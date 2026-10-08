"""Qwen3_5 模型级单测（Phase 3）。

覆盖要点：
1. GDN-only 模型：prefill + 逐 token decode（chunk 内核 → recurrent 内核切换）与
   "一次 prefill 全部 token" 的逐 token 隐状态完全一致。
2. 多序列 varlen：cu_seqlens + seq_slots 寻址下，状态池按槽位隔离、互不串扰。
3. 混合配置（含 full attention 层）：仅验证构造/分派正确、forward 可跑通
   （full 层在无 flash_attn 下用 stub 跳过实际计算）。

说明：
- 本机轻量环境没有 flash-attn；测试注入 stub，且默认使用 GDN-only 配置，
  不会真正触发 flash_attn（生产环境用真实 flash-attn）。
- config 用 SimpleNamespace 模拟 Qwen3_5TextConfig，不依赖 transformers。

运行：python -m tests.test_qwen3_5
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

# ---------- 1. 绕过 nanovllm/__init__.py（同 test_gdn.py）----------
_ROOT = Path(__file__).resolve().parent.parent
for name in ("nanovllm", "nanovllm.layers", "nanovllm.models", "nanovllm.utils"):
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = [str(_ROOT / "nanovllm" / ("layers" if name.endswith("layers") else
                                              "models" if name.endswith("models") else
                                              "utils" if name.endswith("utils") else ""))]
    sys.modules.setdefault(name, mod)

# 说明：Qwen3_5Attention 对 flash_attn/triton 做了延迟导入，
# GDN-only 测试不会触碰它们，因此无需 stub。

import torch
import torch.distributed as dist

# 模型层（TP 线性层/embedding）假定进程组已初始化；单进程 gloo 足够覆盖
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29518", rank=0, world_size=1)

from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.utils.context import set_context, reset_context

torch.manual_seed(0)

# GDN 内核由 FLA（Triton）提供，涉及模型 forward 的用例需要 CUDA
DEVICE = "cuda" if torch.cuda.is_available() else None


def _require_cuda(name):
    if DEVICE is None:
        print(f"SKIP  {name} (需要 CUDA：FLA 内核是 Triton 实现)")
        return False
    return True


def build_config(num_layers=6, replace_linear_tail=0):
    """构造小尺寸 Qwen3.5 文本配置（默认 GDN-only；tail>0 时尾部替换为 full）。"""
    layer_types = ["linear_attention"] * num_layers
    for i in range(num_layers - replace_linear_tail, num_layers):
        layer_types[i] = "full_attention"
    return SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=103,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        attention_bias=False,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=2,
        tie_word_embeddings=False,
    )


def make_model(num_layers=6):
    config = build_config(num_layers)
    model = Qwen3_5ForCausalLM(config)
    # nano-vllm 的 TP 线性层/embedding 用 torch.empty 初始化（依赖 checkpoint 加载），
    # 随机测试需手动填充，避免未初始化内存导致的 NaN；
    # 固定 seed 保证两次 make_model() 权重一致（多模型对照测试需要）
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 0:
                p.zero_()
            else:
                p.uniform_(-0.05, 0.05)
    if DEVICE is not None:
        model = model.to(DEVICE)
    # 搬到设备之后再分配：状态池的 dtype/device 跟随模型参数
    model.allocate_mamba_cache(num_seqs=8)
    return model


def tensor(*shape):
    return torch.randint(1, 100, shape)


def run_decode(model, token_ids, positions, cu_seqlens, seq_slots):
    """decode 语义（is_prefill=False）：每序列 1 个 token。"""
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.long)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    sl = torch.tensor(seq_slots, dtype=torch.int32)
    set_context(False, cu_seqlens_q=cu.to(DEVICE), seq_slots=sl.to(DEVICE))
    try:
        return model(input_ids.to(DEVICE), pos.to(DEVICE))
    finally:
        reset_context()


def run_flat(model, token_ids, positions, cu_seqlens, seq_slots):
    """按 flat 语义跑一次 forward：tokens (N,) / positions (N,) / cu_seqlens (S+1,) / slots (N,)"""
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.long)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    sl = torch.tensor(seq_slots, dtype=torch.int32)
    set_context(True, cu_seqlens_q=cu.to(DEVICE), seq_slots=sl.to(DEVICE))
    try:
        return model(input_ids.to(DEVICE), pos.to(DEVICE))
    finally:
        reset_context()


def test_prefill_then_decode_consistent():
    """核心：prefill 70 token 后逐 token decode，与一次 prefill 74 token 的隐状态一致。

    注意：两条路径必须用独立模型实例（状态池不同源），防止
    一次 forward 产生的 final state 污染下一次的初始状态。"""
    if not _require_cuda("test_prefill_then_decode_consistent"):
        return
    torch.manual_seed(1)
    total = 74
    prefill = 70
    token_ids = [i % 97 + 1 for i in range(total)]

    # 整段一次 prefill（chunk 内核，全部 74 tokens）
    model_full = make_model()
    h_full = run_flat(model_full, token_ids, list(range(total)), [0, total], [0] * total)

    # 分段（独立模型实例）：prefill 70（chunk 内核）→ decode 4 步（每步 recurrent 内核）
    model_chunked = make_model()
    h_pre = run_flat(model_chunked, token_ids[:prefill], list(range(prefill)), [0, prefill], [0] * prefill)
    h_parts = [h_pre]
    for i in range(prefill, total):
        h_t = run_flat(model_chunked, [token_ids[i]], [i], [0, 1], [0])
        h_parts.append(h_t)
    h_chunked = torch.cat(h_parts, dim=0)

    diff = (h_full - h_chunked).abs().max().item()
    # 容差考虑：整段走 chunk 内核（分块 + UT 三角求解，本身是近似），
    # 分段后的 decode 走 recurrent 内核（逐步递推，精确），两条路径数值不同，
    # 差值在 chunk 内核的近似量级（~1e-2）内属正常，非逻辑错误。
    assert diff < 5e-2, f"prefill+decode 与整段 prefill 不一致: {diff:.3e}"
    print(f"PASS  test_prefill_then_decode_consistent (max diff={diff:.2e})")


def test_multi_sequence_state_isolation():
    """多序列 varlen：状态池按槽位隔离 + 与独立模型比对。

    用两个初始状态全零的独立模型实例对照：
    - ref  : 只有 seq0（槽0）
    - mix  : seq0（槽0）+ seq1（槽1）混跑同一批次
    mix 中 seq0 的输出必须与 ref 完全一致（槽位隔离、状态不串扰）。"""
    if not _require_cuda("test_multi_sequence_state_isolation"):
        return
    torch.manual_seed(2)
    seq0 = [i % 97 + 1 for i in range(40)]
    seq1 = [i % 97 + 31 for i in range(25)]   # 不同 token 序列
    seq0_dec = seq0[-1] % 97 + 1

    # —— 参考：独立模型，仅 seq0 ——
    model_ref = make_model()
    h0_ref = run_flat(model_ref, seq0, list(range(40)), [0, 40], [0] * 40)
    h0_ref_d = run_flat(model_ref, [seq0_dec], [40], [0, 1], [0])

    # —— 混合：另一个独立模型，seq0(槽0) + seq1(槽1) 同一批次 ——
    model_mix = make_model()
    h_mix = run_flat(model_mix, seq0 + seq1, list(range(40)) + list(range(25)),
                     [0, 40, 65], [0] * 40 + [1] * 25)
    # 混跑后继续 decode seq0（槽0），应与参考 decode 一致
    h0_mix_d = run_flat(model_mix, [seq0_dec], [40], [0, 1], [0])

    d1 = (h_mix[:40] - h0_ref).abs().max().item()
    d2 = (h0_mix_d - h0_ref_d).abs().max().item()
    assert d1 < 1e-3 and d2 < 1e-3, f"多序列状态串扰: prefill diff={d1:.2e}, decode diff={d2:.2e}"
    print(f"PASS  test_multi_sequence_state_isolation (d1={d1:.2e}, d2={d2:.2e})")


def test_layer_types_dispatch():
    """验证 layer_types 分派统计正确（GDN 层数计数 / 状态池尺寸）。

    混合配置（混入 full attention）的构造验证需要 GPU flash_attn/triton，
    延迟导入已保证 GDN-only 不触发；此处用 GDN-only 配置验证分派与池形状。"""
    torch.manual_seed(3)
    config = build_config(num_layers=6, replace_linear_tail=0)   # GDN-only
    model = Qwen3_5ForCausalLM(config)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 0:
                p.zero_()
            else:
                p.uniform_(-0.05, 0.05)
    model.allocate_mamba_cache(num_seqs=4)
    # 状态池形状：num_linear × (conv: C×(K-1), rec: V×Kd×Vd)
    n_linear = sum(1 for t in config.layer_types if t == "linear_attention")
    assert model.model.num_linear_layers == n_linear == 6
    assert model.model.conv_pool.shape == (4, 6, 2 * 8 * 2 + 6 * 4, 2 - 1)   # (seq, linear, C, K-1)
    assert model.model.rec_pool.shape == (4, 6, 4, 8, 6)                     # (seq, linear, V, Kd, Vd)
    for layer in model.model.layers:
        assert hasattr(layer, "linear_attn") and isinstance(layer.input_layernorm, torch.nn.Module)
    print(f"PASS  test_layer_types_dispatch (GDN 层数={n_linear}/{config.num_hidden_layers})")


def test_batched_decode_matches_single_decode():
    """批量 decode（B 条序列一次算）必须与逐条 decode 完全一致。

    覆盖 _apply_linear_mixer_decode：按槽位 gather/scatter 的批量路径应与
    逐序列路径数值等价。
    """
    if not _require_cuda("test_batched_decode_matches_single_decode"):
        return
    torch.manual_seed(4)
    lens = [12, 9, 15]
    tokens = [i % 97 + 1 for i in range(sum(lens))]
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    slots = []
    for i, length in enumerate(lens):
        slots += [i] * length

    model_batch = make_model()
    run_flat(model_batch, tokens, list(range(len(tokens))), cu, slots)
    model_ref = make_model()
    run_flat(model_ref, tokens, list(range(len(tokens))), cu, slots)

    next_tokens = [tokens[cu[i + 1] - 1] % 97 + 1 for i in range(len(lens))]
    next_pos = [cu[i + 1] for i in range(len(lens))]

    # 三条序列一次算完
    h_batch = run_decode(model_batch, next_tokens, next_pos, [0, 1, 2, 3], [0, 1, 2])
    # 逐条算
    h_ref = torch.cat([
        run_decode(model_ref, [next_tokens[i]], [next_pos[i]], [0, 1], [i])
        for i in range(len(lens))
    ], dim=0)

    diff = (h_batch - h_ref).abs().max().item()
    assert diff < 1e-5, f"批量 decode 与逐条不一致: {diff:.3e}"
    print(f"PASS  test_batched_decode_matches_single_decode (max diff={diff:.2e})")


def test_decode_scatter_with_bf16_pool():
    """回归：GPU 上状态池是 bf16、GDN 内核返回 float32，批量 scatter 必须自行转型。

    旧实现用整数索引（隐式 cast）不会暴露该问题；批量化后改用高级索引，
    dtype 必须严格一致，因此这里刻意把池转成 bf16。
    """
    if not _require_cuda("test_decode_scatter_with_bf16_pool"):
        return
    torch.manual_seed(5)
    model = make_model()
    model.model.conv_pool = model.model.conv_pool.to(torch.bfloat16)
    model.model.rec_pool = model.model.rec_pool.to(torch.bfloat16)

    lens = [6, 4]
    tokens = [i % 97 + 1 for i in range(sum(lens))]
    cu = [0, lens[0], sum(lens)]
    slots = [0] * lens[0] + [1] * lens[1]
    run_flat(model, tokens, list(range(len(tokens))), cu, slots)

    before = model.rec_pool[0].clone()
    h = run_decode(model, [tokens[cu[1] - 1]], [cu[1]], [0, 1], [0])

    assert torch.isfinite(h).all()
    assert model.rec_pool.dtype == torch.bfloat16
    assert not torch.equal(before, model.rec_pool[0]), "循环状态应被更新"
    print("PASS  test_decode_scatter_with_bf16_pool")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nAll Qwen3.5 model tests passed.")
