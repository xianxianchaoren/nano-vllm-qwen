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
    causal_conv1d_varlen,
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll Phase 1/2 tests passed.")
