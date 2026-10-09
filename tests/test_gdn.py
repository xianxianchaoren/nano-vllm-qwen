"""Phase 1 + 2 单测：基础算子与 GDN 内核。

覆盖：
    1. ZeroCenteredRMSNorm / RMSNormGated —— 与手算公式逐值对照
    2. Partial RoPE —— 旋转部分范数不变、pass-through 部分原样透传
    3. short conv —— prefill 全量 vs decode 增量
    4. FLA recurrent vs chunk delta rule —— 数学等价性（输出 + 终态）
    5. FLA delta rule vs 朴素参考实现 —— 语义正确性（按 Eq.1-5 逐步展开）
    6. Qwen3_5GatedDeltaNet —— prefill 全量 vs 分段（decode 续算）一致性

说明：delta rule 内核使用 FLA 的 Triton 实现，因此第 4~6 项需要 CUDA；
无 GPU 时这三项会打印 SKIP，其余（1~3）仍可在 CPU 上跑。

运行：python -m tests.test_gdn  （无需 pytest，纯 assert）
"""
import sys
import types
from itertools import accumulate
from pathlib import Path

# ---------------------------------------------------------------
# 绕过 nanovllm/__init__.py（它 import LLMEngine → transformers/flash_attn，
# 本机轻量环境中没有这些重依赖；Phase 1/2 算子只依赖 torch）。
# 方法：向 sys.modules 注入带 __path__ 的"空壳"包，让子模块从文件直接加载。
# ---------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_nanovllm = types.ModuleType("nanovllm")
_nanovllm.__path__ = [str(_ROOT / "nanovllm")]
_nanovllm.__package__ = "nanovllm"
sys.modules.setdefault("nanovllm", _nanovllm)
_layers = types.ModuleType("nanovllm.layers")
_layers.__path__ = [str(_ROOT / "nanovllm" / "layers")]
_layers.__package__ = "nanovllm.layers"
sys.modules.setdefault("nanovllm.layers", _layers)

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from nanovllm.layers.layernorm import ZeroCenteredRMSNorm, RMSNormGated
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.gated_delta_net import (
    Qwen3_5GatedDeltaNet,
    causal_conv1d_fn,
    causal_conv1d_update,
    causal_conv1d_update_split,
    causal_conv1d_varlen,
    fla_recurrent_decode,
    gdn_recurrent_pool_decode,
    varlen_conv_state,
)

torch.manual_seed(0)

# FLA 内核是 Triton/CUDA 实现，无 GPU 时相关用例跳过
DEVICE = "cuda" if torch.cuda.is_available() else None


def _require_cuda(name):
    if DEVICE is None:
        print(f"SKIP  {name} (需要 CUDA：FLA 内核是 Triton 实现)")
        return False
    return True


def assert_close(a, b, tol=1e-4, msg=""):
    a, b = a.float(), b.float()
    diff = (a - b).abs().max().item()
    assert diff < tol, f"{msg}: max diff = {diff:.3e}"
    return diff


# ======================================================================
# 1. Zero-Centered RMSNorm
# ======================================================================
def test_zero_centered_rmsnorm():
    x = torch.randn(2, 8, 16)
    norm = ZeroCenteredRMSNorm(16, eps=1e-6)

    # 初始（weight=0）时等于标准 RMSNorm：x / rms(x) * 1
    out = norm(x)
    manual = x / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    assert_close(out, manual, 1e-5, "ZeroCenteredRMSNorm 初始退化")

    # 随机 weight 后应等于 x_norm * (1 + w)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16) * 0.1)
    out = norm(x)
    manual = manual * (1 + norm.weight)
    assert_close(out, manual, 1e-5, "ZeroCenteredRMSNorm (1+w)")


def test_rmsnorm_gated():
    x = torch.randn(2, 8, 4)      # (..., head_v_dim)
    gate = torch.randn(2, 8, 4)   # z
    m = RMSNormGated(4, eps=1e-6)
    out = m(x, gate)
    # Qwen3.5 的 RMSNormGated 用 silu 作为 gate 激活（Flash-Next 才改为 sigmoid）
    manual = (x / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)) * torch.nn.functional.silu(gate.float())
    assert_close(out, manual, 1e-5, "RMSNormGated 应为 rms(x)·silu(z)")


# ======================================================================
# 2. Partial RoPE
# ======================================================================
def test_partial_rope():
    head_dim, rotary_dim, max_pos = 16, 8, 256   # 只旋转前 8 维（50%）
    rope = get_rope(head_dim, rotary_dim=rotary_dim, max_position=max_pos, base=10000.0)
    positions = torch.tensor([0, 5, 100])
    q = torch.randn(3, 4, head_dim)              # (seq, heads, dim)
    k = torch.randn(3, 4, head_dim)

    qr, kr = rope(positions, q, k)

    # (a) 旋转保持范数不变（旋转是正交变换）
    assert_close(qr.norm(dim=-1), q.norm(dim=-1), 1e-5, "RoPE 范数保持")
    # (b) pass-through 部分（后 head_dim-rotary_dim 维）必须原样不变
    assert_close(qr[..., rotary_dim:], q[..., rotary_dim:], 1e-6, "pass-through")
    # (c) 位置 0 不旋转（cos=1, sin=0）
    assert_close(qr[0], q[0], 1e-6, "position 0 不旋转")
    # (d) 兼容全旋转（旧行为）：rotary_dim == head_dim
    rope_full = get_rope(head_dim, rotary_dim=head_dim, max_position=max_pos, base=10000.0)
    qr_full, _ = rope_full(positions, q, k)
    assert qr_full.shape == q.shape


# ======================================================================
# 3. short conv：prefill 全量 vs decode 增量
# ======================================================================
def test_conv1d_incremental():
    C, K, L = 8, 4, 20
    weight = torch.randn(C, K)   # 约定：每通道 1D 卷积核，形状 (C, K)
    x = torch.randn(2, C, L)

    # 全量（含 silu）——注意 return_state=False 返回单个 tensor，不可解包！
    y_full = causal_conv1d_fn(x, weight, activation="silu", return_state=False)
    # 用前 L-1 个 token 的全量调用得到 state = x[L-K+1 .. L-2]（供"下一个"token 使用）
    _, state = causal_conv1d_fn(x[:, :, :-1], weight, activation="silu", return_state=True)
    assert state.shape == (2, C, K - 1)

    # 增量：state 拼上新 token x[L-1] → 输出应等于全量的最后一个位置
    last = causal_conv1d_update(x[:, :, -1:], state.clone(), weight, activation="silu")
    assert_close(y_full[..., -1:], last, 1e-5, "conv 增量输出")


def test_conv1d_varlen_matches_dense():
    """多序列拼接（varlen）卷积必须与逐序列卷积完全一致。

    覆盖 causal_conv1d_varlen 的跨序列边界修正，以及 varlen_conv_state 的
    末 pad 个输入提取（含段长不足 pad 时的右对齐补零）。
    """
    torch.manual_seed(11)
    C, K = 6, 4
    lens = [5, 8, 2]                       # 最后一段短于 pad = K-1 = 3
    tot = sum(lens)
    weight = torch.randn(C, K)
    bias = torch.randn(C)
    x = torch.randn(1, C, tot)
    cu = torch.tensor([0] + list(accumulate(lens)), dtype=torch.int32)

    got = causal_conv1d_varlen(x, weight, bias, activation="silu", cu_seqlens=cu)
    ref = torch.cat([
        causal_conv1d_fn(x[:, :, cu[i]:cu[i + 1]], weight, bias,
                         activation="silu", return_state=False)
        for i in range(len(lens))
    ], dim=-1)
    assert_close(got, ref, 1e-5, "varlen conv vs 逐序列 conv")

    # conv_state：每段最后 pad 个输入；段长不足时左侧补零
    pad = K - 1
    state = varlen_conv_state(x, cu, pad)
    ref_state = torch.zeros(len(lens), C, pad)
    for i in range(len(lens)):
        seg = x[0, :, cu[i]:cu[i + 1]]
        n = min(pad, seg.shape[-1])
        ref_state[i, :, pad - n:] = seg[:, -n:]
    assert_close(state, ref_state, 1e-6, "varlen conv_state")


# ======================================================================
# 4. FLA recurrent vs FLA chunk（数学等价，均需 CUDA）
# ======================================================================
def test_recurrent_chunk_equivalence():
    if not _require_cuda("test_recurrent_chunk_equivalence"):
        return
    batch, seq, nk, nv, kd, vd = 2, 130, 4, 16, 8, 6   # 130 故意不整除 chunk 64
    # FLA 约定：q/k 需已按 GQA 扩展为 num_v_heads（层内由 repeat_interleave 完成）
    q = torch.randn(batch, seq, nk, kd, device=DEVICE).repeat_interleave(nv // nk, dim=2)
    k = torch.randn(batch, seq, nk, kd, device=DEVICE).repeat_interleave(nv // nk, dim=2)
    v = torch.randn(batch, seq, nv, vd, device=DEVICE)
    g = -torch.rand(batch, seq, nv, device=DEVICE).abs() * 0.5   # log 衰减 ≤ 0
    beta = torch.sigmoid(torch.randn(batch, seq, nv, device=DEVICE))

    out_r, state_r = fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    out_c, state_c = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)

    assert_close(out_r, out_c, 1e-2, "FLA recurrent vs chunk 输出")
    assert_close(state_r, state_c, 1e-2, "FLA recurrent vs chunk 终态")

    # 从给定 initial_state 续算也应一致
    init = torch.randn(batch, nv, kd, vd, device=DEVICE)
    out_r2, _ = fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=init, output_final_state=False,
        use_qk_l2norm_in_kernel=True)
    out_c2, _ = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=init, output_final_state=False,
        use_qk_l2norm_in_kernel=True)
    assert_close(out_r2, out_c2, 1e-2, "带 initial_state 的一致性")


# ======================================================================
# 5. FLA delta rule vs 朴素参考实现（严格按 Eq.1-5 展开，需 CUDA）
# ======================================================================
def test_delta_rule_vs_naive():
    if not _require_cuda("test_delta_rule_vs_naive"):
        return
    batch, seq, nv, kd, vd = 1, 6, 2, 4, 3
    torch.manual_seed(7)
    # scale=1.0 且不在内核内做 l2norm：把 delta rule 语义与缩放/归一化约定解耦，
    # 朴素实现只需按 Eq.1-5 逐步展开即可对齐
    q = torch.randn(batch, seq, nv, kd, device=DEVICE)
    k = torch.randn(batch, seq, nv, kd, device=DEVICE)
    v = torch.randn(batch, seq, nv, vd, device=DEVICE)
    g = -torch.rand(batch, seq, nv, device=DEVICE) * 0.3
    beta = torch.sigmoid(torch.randn(batch, seq, nv, device=DEVICE))

    out, _ = fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, scale=1.0, output_final_state=False,
        use_qk_l2norm_in_kernel=False)

    # 朴素实现：按论文 Eq.1-5（张量布局 (B, L, H, D)，索引 [:, i, :]）
    S = torch.zeros(batch, nv, kd, vd, device=DEVICE)
    expected = torch.zeros(batch, nv, seq, vd, device=DEVICE)   # (B, H, L, D)
    for i in range(seq):
        alpha = g[:, i, :].exp()                                # 衰减率 α_t
        S = S * alpha[..., None, None]                          # Eq.1 衰减
        pred = (S * k[:, i, :].unsqueeze(-1)).sum(dim=-2)       # Sᵀk
        err = (v[:, i, :] - pred) * beta[:, i, :].unsqueeze(-1) # Eq.2 误差
        S = S + k[:, i, :].unsqueeze(-1) * err.unsqueeze(-2)    # Eq.3 写入
        expected[:, :, i] = (S * q[:, i, :].unsqueeze(-1)).sum(dim=-2)  # Eq.4 读出
    expected = expected.transpose(1, 2)
    assert_close(out, expected, 1e-4, "FLA recurrent 与朴素 Eq.1-5 一致")


# ======================================================================
# 6. GDN 层：prefill（chunk）与 分段续算（recurrent）输出一致
# ======================================================================
def test_gdn_layer_prefill_vs_chunked():
    if not _require_cuda("test_gdn_layer_prefill_vs_chunked"):
        return
    torch.manual_seed(42)
    B, L, D = 2, 70, 32   # L 不整除 64
    model = Qwen3_5GatedDeltaNet(
        hidden_size=D,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=4,
    ).to(DEVICE)
    # FusedInputProj / TP 线性层用 torch.empty 初始化（正式路径依赖 checkpoint 加载），
    # 随机单测必须自己填充，否则会读到未初始化内存（NaN/垃圾），测试结果不可复现。
    # A_log / dt_bias 保留模块自身的初始化（保证 g = -exp(A)·softplus(a + dt) 的量级合理）。
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "A_log" in name or "dt_bias" in name:
                continue
            p.uniform_(-0.05, 0.05)
    x = torch.randn(B, L, D, device=DEVICE)

    # (a) 一次 prefill
    out_full, conv_state_full, rec_state_full = model(x)

    # (b) 分段：prefill 前 64 token（chunk 内核）→ decode 剩余 6 token（每步 recurrent 增量）
    out_chunked_a, conv_s, rec_s = model(x[:, :64])
    outs_a = [out_chunked_a]
    for i in range(64, L):
        o, conv_s, rec_s = model(x[:, i:i + 1], conv_s, rec_s)
        outs_a.append(o)
    out_chunked = torch.cat(outs_a, dim=1)

    assert_close(out_full, out_chunked, 1e-2, "GDN prefill vs 分段续算")
    assert_close(conv_state_full, conv_s, 1e-4, "卷积状态一致")
    assert_close(rec_state_full, rec_s, 1e-2, "循环状态一致")


def test_conv1d_update_split():
    """decode 融合 conv 必须与"独立参考 + 逐算子实现"逐值一致。

    CPU 上走 fallback（逐算子 torch 实现）；CUDA 上走融合 Triton 内核，并把
    [q|k|v] 三段直接写成连续缓冲区。两条路径都要对上同一个参考，才能保证
    "换内核"不改变语义、布局变化也不影响下游。
    """
    torch.manual_seed(13)
    B, key_dim, value_dim, K = 2, 2, 4, 4
    conv_dim = 2 * key_dim + value_dim
    dev = DEVICE or "cpu"
    weight = torch.randn(conv_dim, K, device=dev)
    bias = torch.randn(conv_dim, device=dev)
    x = torch.randn(B, conv_dim, 1, device=dev)
    state = torch.randn(B, conv_dim, K - 1, device=dev)

    # 参考 1：独立公式 —— 窗口 = [state | x]（长度恰好 K），逐通道加权求和 → bias → silu
    win = torch.cat([state, x], dim=-1)
    ref = torch.nn.functional.silu((win * weight.unsqueeze(0)).sum(-1) + bias)
    ref_q, ref_k, ref_v = ref.split([key_dim, key_dim, value_dim], dim=-1)
    # 参考 2：逐算子实现（语义相同、实现不同）
    state_ref = state.clone()
    out_ref = causal_conv1d_update(x, state_ref, weight, bias, "silu")
    op_q, op_k, op_v = torch.split(
        out_ref.transpose(1, 2), [key_dim, key_dim, value_dim], dim=-1)

    state_new = state.clone()
    q, k, v = causal_conv1d_update_split(x, state_new, weight, bias, key_dim, "silu")

    for name, got, exp in (("q", q, ref_q), ("k", k, ref_k), ("v", v, ref_v)):
        assert got.shape == (B, 1, exp.shape[-1]), f"{name} 形状不符: {got.shape}"
        assert_close(got.squeeze(1), exp, 1e-5, f"融合 conv {name} vs 独立参考")
    assert_close(q.squeeze(1), op_q.squeeze(1), 1e-5, "融合 conv q vs 逐算子实现")
    assert_close(k.squeeze(1), op_k.squeeze(1), 1e-5, "融合 conv k vs 逐算子实现")
    assert_close(v.squeeze(1), op_v.squeeze(1), 1e-5, "融合 conv v vs 逐算子实现")
    # conv_state 原地左移一位、末位写当前输入（= 窗口的最后 state_len 项）
    assert_close(state_new, win[..., -(K - 1):], 1e-6, "conv_state 移位语义")
    assert_close(state_new, state_ref, 1e-6, "conv_state vs 逐算子实现")

    # 无 bias（Qwen3.5 实际配置：conv bias=False）
    state_nb = state.clone()
    q_nb, k_nb, v_nb = causal_conv1d_update_split(
        x, state_nb, weight, None, key_dim, "silu")
    ref_nb = torch.nn.functional.silu((win * weight.unsqueeze(0)).sum(-1))
    nb_q, nb_k, nb_v = ref_nb.split([key_dim, key_dim, value_dim], dim=-1)
    assert_close(q_nb.squeeze(1), nb_q, 1e-5, "无 bias 时融合 conv q")
    assert_close(k_nb.squeeze(1), nb_k, 1e-5, "无 bias 时融合 conv k")
    assert_close(v_nb.squeeze(1), nb_v, 1e-5, "无 bias 时融合 conv v")

    if DEVICE is not None:
        # 融合路径必须真的被走到，且产物是连续缓冲区（下游 FLA 直接吃，免拷贝）
        from nanovllm.layers.triton_launch import _LAUNCHERS
        assert any(kk[0] == "conv_update_split" for kk in _LAUNCHERS), \
            "融合 conv 内核未被使用"
        for name, t in (("q", q), ("k", k), ("v", v)):
            assert t.is_contiguous(), f"融合 conv 的 {name} 必须是连续缓冲区"


def test_fla_recurrent_bound_launcher():
    """预绑定启动器必须与 FLA 公开 API 逐值一致（同一 kernel、同一组 constexpr）。

    额外覆盖：非连续输入必须**回退**公开 API（内核没有 stride 入参，预绑定路径
    绕开了 input_guard 的 .contiguous()，带错误布局发射会静默算错 batch 维）。
    """
    if not _require_cuda("test_fla_recurrent_bound_launcher"):
        return
    torch.manual_seed(17)
    B, nk, nv, kd, vd = 4, 2, 4, 8, 6
    q = torch.randn(B, 1, nk, kd, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, 1, nk, kd, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, 1, nv, vd, device=DEVICE, dtype=torch.bfloat16)
    # 门控：g 是 log 空间（∈ (−∞,0]）的 float32，beta 已过 sigmoid
    g = -torch.rand(B, 1, nv, device=DEVICE).abs() * 0.5
    beta = torch.sigmoid(torch.randn(B, 1, nv, device=DEVICE, dtype=torch.bfloat16))
    h0 = torch.randn(B, nv, kd, vd, device=DEVICE)

    ref_out, ref_state = fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta,
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    out, state = fla_recurrent_decode(
        q, k, v, g, beta, h0,
        fallback=lambda: fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta,
            initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True))

    assert out.shape == ref_out.shape and state.shape == ref_state.shape
    assert_close(out, ref_out, 1e-6, "预绑定启动器输出 vs FLA 公开 API")
    assert_close(state, ref_state, 1e-6, "预绑定启动器终态 vs FLA 公开 API")

    from nanovllm.layers.triton_launch import _LAUNCHERS
    assert any(kk[0] == "fla_recurrent_decode" for kk in _LAUNCHERS), \
        "FLA 预绑定启动器未被使用（可能触发了回退）"

    # 非连续 g/beta（模拟 in_proj_ba 的 [b|a] 半区视图）必须回退而不是算错
    ba = torch.randn(B, 1, 2 * nv, device=DEVICE)
    g_nc, beta_nc = ba[..., :nv], ba[..., nv:]
    assert not g_nc.is_contiguous() and not beta_nc.is_contiguous()
    hit = []

    def fallback():
        hit.append(1)
        return fused_recurrent_gated_delta_rule(
            q, k, v, g=g_nc, beta=beta_nc,
            initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True)

    out_nc, state_nc = fla_recurrent_decode(q, k, v, g_nc, beta_nc, h0,
                                            fallback=fallback)
    assert hit, "非连续输入没有回退到公开 API"
    ref_nc = fused_recurrent_gated_delta_rule(
        q, k, v, g=g_nc.contiguous(), beta=beta_nc.contiguous(),
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    assert_close(out_nc, ref_nc[0], 1e-6, "回退路径输出")


def test_pool_decode_matches_gather_path():
    """池内原地读写的 recurrent 内核 vs gather + FLA 路径：输出与池内容都要一致。"""
    if not _require_cuda("test_pool_decode_matches_gather_path"):
        return
    torch.manual_seed(23)
    B, nk, nv, kd, vd = 4, 2, 4, 8, 6
    SLOTS, LAYERS = 6, 3
    q = torch.randn(B, 1, nk, kd, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, 1, nk, kd, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, 1, nv, vd, device=DEVICE, dtype=torch.bfloat16)
    g = -torch.rand(B, 1, nv, device=DEVICE).abs() * 0.5
    beta = torch.sigmoid(torch.randn(B, 1, nv, device=DEVICE, dtype=torch.bfloat16))
    pool0 = torch.randn(SLOTS, LAYERS, nv, kd, vd, device=DEVICE)
    slots = torch.tensor([0, 3, 4, 5], dtype=torch.int32, device=DEVICE)

    # 逐个层下标验证：必须**逐位**相等（layer_offset 写错时只会在 li != 1 暴露）
    for li in (0, 1, 2):
        pool_ref = pool0.clone()
        rec = pool_ref[slots.long(), li]
        out_ref, new_rec = fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=rec, output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        pool_ref[slots.long(), li] = new_rec

        pool_fused = pool0.clone()
        out = gdn_recurrent_pool_decode(q, k, v, g, beta, pool_fused, slots, li)

        assert out.shape == out_ref.shape
        assert torch.equal(out.reshape_as(out_ref), out_ref), \
            f"li={li}: 池内 recurrent 输出与原路径不是逐位相等"
        assert torch.equal(pool_fused[slots.long(), li], pool_ref[slots.long(), li]), \
            f"li={li}: 池内状态更新与原路径不是逐位相等"
        # 其它层、其它槽位都不能被写坏
        others = [l for l in range(LAYERS) if l != li]
        assert torch.equal(pool_fused[:, others], pool0[:, others]), \
            f"li={li}: 其它层被写坏"
        unused = [i for i in range(SLOTS) if i not in slots.tolist()]
        assert torch.equal(pool_fused[unused], pool0[unused]), \
            f"li={li}: 未使用的槽位被写坏"

    from nanovllm.layers.triton_launch import _LAUNCHERS
    assert any(kk[0] == "gdn_rec_pool" for kk in _LAUNCHERS), \
        "池内 recurrent 内核未被使用"


def test_bound_launch_arg_alignment():
    """预绑定启动器的位置参数必须与 Triton 内核签名逐一对齐（无需 GPU）。

    Triton 按"签名下标"取位置参数，并把 None / int(1) 特化成 constexpr；
    参数一旦错位不会报错，只会静默算错 —— 而预绑定恰好绕开了 binder 的参数
    校验。这里把 bound_launch 换成记录器，再用 Triton 自己的特化函数核对
    每个 (kernel, args, constexprs) 组合，把这“无 GPU 也测不到”的风险锁住。
    """
    from triton.runtime.jit import native_specialize_impl
    from nanovllm.layers import gated_delta_net as gdn_mod
    from nanovllm.layers import layernorm as ln_mod
    from nanovllm.layers import triton_launch as tl_mod

    recorded = []

    class _StubRunner:
        def __call__(self, *a):
            pass

    def recorder(jit_fn, tag, key, grid, *args, constexprs, num_warps, num_stages=3):
        recorded.append((tag, jit_fn, args, dict(constexprs)))
        return _StubRunner(), args

    class _StubBackend:
        def get_tensor_specialization(self, *a, **kw):
            t = a[0]
            ty = str(t.dtype).replace("torch.", "")
            ok = all(s % 16 == 0 for s in t.stride()) and t.data_ptr() % 16 == 0
            return ("*" + ty, "D" if ok else "N")

    saved = (tl_mod.bound_launch, ln_mod.bound_launch, gdn_mod.bound_launch)
    tl_mod.bound_launch = recorder
    ln_mod.bound_launch = recorder
    gdn_mod.bound_launch = recorder
    try:
        x = torch.randn(16, 64, dtype=torch.bfloat16)
        w = torch.zeros(64, dtype=torch.bfloat16)
        gate = torch.randn(16, 64, dtype=torch.bfloat16)
        ln_mod.rms_norm(x, w, 1e-6, zero_centered=True)
        ln_mod.rms_norm_gated(x, w, gate, 1e-6)

        B, key_dim, value_dim, K = 3, 8, 16, 4
        conv_dim = 2 * key_dim + value_dim
        xs = torch.randn(B, conv_dim, dtype=torch.bfloat16)
        query = torch.empty(B, 1, key_dim, dtype=torch.bfloat16)
        key = torch.empty(B, 1, key_dim, dtype=torch.bfloat16)
        value = torch.empty(B, 1, value_dim, dtype=torch.bfloat16)
        for bias in (None, torch.randn(conv_dim)):
            gdn_mod._conv_update_split_fused(
                xs, torch.randn(conv_dim, K), bias,
                torch.zeros(B, conv_dim, K - 1), query, key, value,
                key_dim, value_dim, K, True)

        b, nk, nv, kd, vd = 2, 2, 4, 8, 6
        ba = torch.randn(b, 1, 2 * nv, dtype=torch.bfloat16)
        gdn_mod.gdn_gate(ba, torch.randn(nv), torch.randn(nv))
        gdn_mod.fla_recurrent_decode(
            torch.randn(b, 1, nk, kd, dtype=torch.bfloat16),
            torch.randn(b, 1, nk, kd, dtype=torch.bfloat16),
            torch.randn(b, 1, nv, vd, dtype=torch.bfloat16),
            -torch.rand(b, 1, nv).abs(), torch.rand(b, 1, nv),
            torch.zeros(b, nv, kd, vd), fallback=lambda: None)
    finally:
        tl_mod.bound_launch, ln_mod.bound_launch, gdn_mod.bound_launch = saved

    # rms_norm / rms_norm_gated / gdn_gate / 融合 conv ×2 / FLA recurrent
    assert len(recorded) == 6, f"记录到的启动次数不对: {len(recorded)}"
    for tag, jit_fn, args, constexprs in recorded:
        base = jit_fn
        while not hasattr(base, "params"):
            base = base.fn
        assert len(args) <= len(base.params), f"{tag}: 位置参数多于形参"
        for i, p in enumerate(base.params):
            if i < len(args):
                assert not p.is_constexpr, \
                    f"{tag}: pos {i} ({p.name}) 是 tl.constexpr 却在位置参数里"
                spec = native_specialize_impl(
                    _StubBackend(), args[i], bool(getattr(p, "is_const", False)),
                    not bool(getattr(p, "do_not_specialize", False)),
                    not bool(getattr(p, "do_not_specialize_on_alignment", False)))
                if spec[0] == "constexpr":
                    # None 占位是预期的；其它值被隐式 constexpr 会成为特化隐患
                    assert args[i] is None, \
                        f"{tag}: pos {i} ({p.name}) 意外被特化成 constexpr: {args[i]!r}"
            else:
                assert p.is_constexpr or p.name in constexprs, \
                    f"{tag}: pos {i} ({p.name}) 缺少实参"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll Phase 1/2 tests passed.")
