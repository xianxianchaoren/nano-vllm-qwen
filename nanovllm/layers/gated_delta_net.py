"""Gated Delta Net（GDN）——Qwen3.5 / Qwen3.8-27B 的线性注意力层核心。

原理综述
========
GDN 是 Qwen3.5/3.8 混合注意力架构中 3/4 层使用的 token mixer
（其余 1/4 层为 full attention）。它把"历史上下文"压缩进一个
**固定大小的循环状态 S**（形状 [num_v_heads, k_dim, v_dim]），
因此 KV 缓存不再随上下文长度增长——这是它支撑 262K 上下文的根本原因。

核心递推（每 token 一步，按 Qwen3.8 论文 Eq.1-5）：

    S̃_{t-1} = α_t · S_{t-1}                  (1) 衰减：决定旧记忆的遗忘速度
    e_t      = v_t − S̃_{t-1}ᵀ · k_t          (2) 误差：旧状态对当前 k 的"预测残差"
    S_t      = S̃_{t-1} + β_t · k_t · e_tᵀ    (3) 写入：只把预测不出的部分写进记忆
    y_t      = S_tᵀ · q_t                     (4) 读出：用当前 query 检索记忆

其中两个数据相关的门控：
    β_t = σ(W_β x_t)                          写入强度（0=不写，1=覆盖）
    α_t = exp[−exp(A)·softplus(W_α x_t + b)]  衰减率（恒在 (0,1)，log 空间参数化）

等价矩阵形式（论文 Eq.5）：
    S_t = α_t·(I − β_t·k_t·k_tᵀ)·S_{t-1} + β_t·k_t·v_tᵀ

Delta rule 的关键 insight：普通线性注意力会"无界累积"外积 k⊗v，
而 delta rule 先减去旧状态已经能预测的部分（S̃·k），只写误差项，
因此重复/相似的 key 是"更新既有记忆"而非"叠加噪声"。

delta rule 内核直接使用 FLA（flash-linear-attention）的 Triton 实现：
    - fused_recurrent_gated_delta_rule：逐 token 递推，用于 decode（单步）
    - chunk_gated_delta_rule          ：按 chunk 并行，用于 prefill（长序列）
两者数学等价；FLA 同时支持 varlen、GVA（分组 value 注意力）与 in-kernel L2 归一化，
也是 vLLM 在非 Hopper/Blackwell 平台上的默认 GDN 后端（Triton/FLA）。

本文件保留 GDN 层的其余部分：Qwen3_5GatedDeltaNet（投影 + short conv + 门控 + 输出）。

参考实现：transformers modeling_qwen3_5.py（Apache 2.0）、fla.ops.gated_delta_rule。
"""
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.utils.op import exp as _fla_exp

try:
    # FLA recurrent 内核本体（用于预绑定启动器，省掉每次调用的 binder 开销）
    from fla.ops.gated_delta_rule.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd_kernel,
    )
except ImportError:   # pragma: no cover - 兼容内核改名/搬迁的 FLA 版本
    fused_recurrent_gated_delta_rule_fwd_kernel = None

from nanovllm.layers.layernorm import RMSNormGated
from nanovllm.layers.triton_launch import bound_launch


# ======================================================================
# GDN 门控融合内核（对应 vLLM 的 fused gating）
# ======================================================================
@triton.jit
def _gdn_gate_kernel(
    ba_ptr, a_log_ptr, dt_bias_ptr, g_ptr, beta_ptr,
    total, H: tl.constexpr, BLOCK: tl.constexpr,
):
    """一次算完 GDN 的两个数据相关门控，取代原来 ~8 个 eager 逐元素 kernel。

    输入 ba = in_proj_ba(x)，形状 (N, 2H)：**前 H 列是 b（写入强度候选）、
    后 H 列是 a（衰减率候选）**（与权重加载顺序一致）。

        beta = sigmoid(b)                                   ∈ (0, 1)
        g    = -exp(A_log) * softplus(a + dt_bias)          ∈ (-∞, 0]

    softplus 用数值稳定形式 max(x,0) + log1p(exp(-|x|))，与 F.softplus 等价
    （阈值两侧都一致）。g 以 float32 输出（FLA 的 chunk/recurrent 内核要求），
    beta 的 dtype 跟随输入（bf16）。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    row = offs // H
    h = offs % H
    base = row * (2 * H)
    b = tl.load(ba_ptr + base + h, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(ba_ptr + base + H + h, mask=mask, other=0.0).to(tl.float32)
    a_log = tl.load(a_log_ptr + h, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h, mask=mask, other=0.0).to(tl.float32)
    x = a + dt_bias
    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    g = -tl.exp(a_log) * softplus
    beta = 1.0 / (1.0 + tl.exp(-b))
    tl.store(g_ptr + offs, g, mask=mask)
    tl.store(beta_ptr + offs, beta.to(beta_ptr.dtype.element_ty), mask=mask)


def gdn_gate(ba: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor):
    """融合门控：返回 (beta, g)，形状均与 ba 的前半部分相同 (..., H)。

    beta 为输入 dtype，g 为 float32（log 空间，∈ (−∞, 0]）。

    产物是**新分配且连续**的张量 —— 这一点很重要：ba 的 [b|a] 两个半区都是行
    stride = 2H 的非连续视图，而 FLA 的 recurrent 内核读 g/beta 时写死了行内
    布局、没有 stride 入参，必须喂给它连续的张量。
    """
    *lead, two_h = ba.shape
    H = two_h // 2
    ba2 = ba.reshape(-1, two_h)
    rows = ba2.shape[0]
    total = rows * H
    beta = torch.empty((rows, H), dtype=ba.dtype, device=ba.device)
    g = torch.empty((rows, H), dtype=torch.float32, device=ba.device)
    BLOCK = 256
    runner, args = bound_launch(
        _gdn_gate_kernel, "gdn_gate",
        (rows, total, H, BLOCK, ba.dtype, a_log.dtype, dt_bias.dtype),
        (triton.cdiv(total, BLOCK),),
        ba2, a_log, dt_bias, g, beta, total,
        constexprs=dict(H=H, BLOCK=BLOCK), num_warps=4,
    )
    runner(*args)
    return beta.reshape(*lead, H), g.reshape(*lead, H)


def gdn_gate_native(ba: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor):
    """CPU / 非连续输入时的参考实现（与 transformers 逐行等价）。"""
    H = ba.shape[-1] // 2
    b, a = torch.split(ba, [H, H], dim=-1)
    beta = b.sigmoid()
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)
    return beta, g


# ======================================================================
# 短因果卷积（Short Conv）：GDN 的局部归纳偏置
# ======================================================================
def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "",
    return_state: bool = False,
    **kwargs,
) -> torch.Tensor:
    """对输入做 depthwise 因果卷积（完整序列一次算完，用于 prefill）。

    形状约定：输入 (B, C, L)，权重 (C, 1, K)（kernel=K）。
    padding 取 K-1 且结果裁回前 L 个位置 → 位置 t 只依赖 t-K+1..t，
    即"因果"（未来信息不泄漏）。这是 Mamba 系列 Short Conv 的标准做法。

    Args:
        hidden_states: (batch, channels, seq_len)
        weight: (channels, 1, kernel_size)  —— 每通道独立的 1D 卷积核
        activation: 卷积后作用的激活函数（Qwen3.5 用 silu）
        return_state: 为 True 时额外返回"卷积前输入"的最近 K-1 个值
            （形状 (B, C, K-1)），供 decode 增量卷积续算使用。
    """
    _, _, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_states.shape[1],   # depthwise：每通道独立卷积
    )[:, :, :seq_len]
    if activation:
        out = F.silu(out)
    out = out.to(hidden_states.dtype)
    if return_state:
        # 卷积因果依赖 t-K+1..t，将来算 t+1 只需最近 K-1 个输入
        # 注意：必须 clone —— 输入可能来自 torch.split 之类的"多视图"结果，
        # 那种视图不允许后续原地 copy_（decode 增量更新会写它）
        conv_state = hidden_states[:, :, -padding:].clone()
        return out, conv_state
    return out


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "",
) -> torch.Tensor:
    """单步（decode）增量卷积：利用缓存的 conv_state 只算新位置。

    conv_state 形状 (batch, channels, K-1)：最近 K-1 个历史值。
    把新 token 拼到历史后做一次无 padding 卷积（长度恰好为 1），
    并**原地更新** conv_state（把最旧的值挤出去）。

    这样 decode 阶段无需对整段历史重算卷积，代价是 O(C·K) 而非 O(C·L)。
    """
    B, C, _ = hidden_states.shape
    state_len = conv_state.shape[-1]
    # 拼接 [历史 K-1 值 | 新值] → 长度 K，一次卷积产出 1 个位置
    hidden_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_new[:, :, -state_len:])   # 腾出位置给新 token
    out = F.conv1d(
        hidden_new, weight.unsqueeze(1), bias, padding=0, groups=C,
    )
    if activation:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


@triton.jit
def _causal_conv1d_update_split_kernel(
    x_ptr, w_ptr, bias_ptr, state_ptr,
    q_ptr, k_ptr, v_ptr,
    stride_xb, stride_sb, stride_qb, stride_kb, stride_vb,
    state_len,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATE: tl.constexpr,
):
    """decode 单步融合内核：因果状态移位 + depthwise 卷积 + 激活 + 按 q/k/v 分段写出。

    每个 program 处理一个 batch 上的一段通道：

    1. 卷积窗口 = conv_state 的最后 K-1 项 + 当前输入（因果性由"只取历史末尾"保证，
       等价于先 cat 再 padding=0 卷积取最后一个输出位置）；
    2. 输出按通道归属分别写进 q / k / v 三块独立缓冲区 —— 因此产物天然连续，
       下游 FLA 内核不再需要对 split 出来的 q/k/v 做 `.contiguous()` 拷贝
       （通道布局是连续的 [q | k | v]，所以每段都在同一个 program 内）；
    3. conv_state 原地左移一位，末端写入当前输入（与参考实现的
       `conv_state.copy_(hidden_new[..., -state_len:])` 一致）。

    相比 torch 参考实现（`torch.cat` + `F.conv1d` + `F.silu` + `copy_` 共 4 次
    算子派发），这里只发一次 kernel。
    """
    pid_c = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    total = 2 * KEY_DIM + VALUE_DIM
    mask = offs < total

    # ---- 1) 卷积：先算当前输入项，再累加 state 的最近 K-1 项 ----
    xn = tl.load(x_ptr + pid_b * stride_xb + offs, mask=mask, other=0.0).to(tl.float32)
    base_s = pid_b * stride_sb + offs * state_len
    acc = xn * tl.load(w_ptr + offs * K + (K - 1), mask=mask, other=0.0).to(tl.float32)
    for j in tl.static_range(K - 1):
        col = state_len - (K - 1) + j
        sv = tl.load(state_ptr + base_s + col, mask=mask, other=0.0).to(tl.float32)
        wj = tl.load(w_ptr + offs * K + j, mask=mask, other=0.0).to(tl.float32)
        acc += wj * sv
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if ACTIVATE:
        acc = acc * tl.sigmoid(acc)          # silu

    # ---- 2) 分三段写出（q / k / v 各自连续）----
    tl.store(q_ptr + pid_b * stride_qb + offs, acc,
             mask=mask & (offs < KEY_DIM))
    tl.store(k_ptr + pid_b * stride_kb + (offs - KEY_DIM), acc,
             mask=mask & (offs >= KEY_DIM) & (offs < 2 * KEY_DIM))
    tl.store(v_ptr + pid_b * stride_vb + (offs - 2 * KEY_DIM), acc,
             mask=mask & (offs >= 2 * KEY_DIM))

    # ---- 3) 状态左移一位：new_state[j] = old[j+1]，末位写当前输入 ----
    ks = tl.arange(0, BLOCK_K)
    shifted = tl.load(state_ptr + base_s[:, None] + (ks + 1)[None, :],
                      mask=mask[:, None] & ((ks + 1) < state_len)[None, :], other=0.0)
    new_s = tl.where((ks == state_len - 1)[None, :], xn[:, None], shifted)
    tl.store(state_ptr + base_s[:, None] + ks[None, :], new_s,
             mask=mask[:, None] & (ks < state_len)[None, :])


def causal_conv1d_update_split(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    key_dim: int,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """decode 单步：融合 conv 更新，并直接把 [q|k|v] 三段写成三块连续缓冲区。

    返回 (query, key, value)，形状均为 (B, 1, dim)；`conv_state` 被原地更新。

    走融合内核的条件（不满足则退回逐算子实现，保证 CPU / 非标准场景可用）：
        - CUDA + 最后一维连续 + `state_len == K - 1`（单步增量卷积的前提）
    """
    batch, conv_dim, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    kernel_size = weight.shape[-1]
    value_dim = conv_dim - 2 * key_dim
    # 通道步长要看二维视图：输入是 (B, C, 1)，转置后末维 size=1、stride 并不为 1
    x = hidden_states[:, :, 0]                        # (B, C)
    fused = (
        seq_len == 1
        and hidden_states.is_cuda
        and x.stride(-1) == 1
        and state_len == kernel_size - 1
        and conv_state.is_contiguous()
        and weight.is_contiguous()
        and (bias is None or bias.is_contiguous())
        and key_dim > 0 and value_dim > 0
    )
    if not fused:
        out = causal_conv1d_update(hidden_states, conv_state, weight, bias, activation)
        # (B, C, L) -> (B, L, C) 后再按通道切段，保证与融合路径同样的输出布局
        return torch.split(out.transpose(1, 2), [key_dim, key_dim, value_dim], dim=-1)

    dtype, device = hidden_states.dtype, hidden_states.device
    query = torch.empty(batch, 1, key_dim, dtype=dtype, device=device)
    key = torch.empty(batch, 1, key_dim, dtype=dtype, device=device)
    value = torch.empty(batch, 1, value_dim, dtype=dtype, device=device)
    _conv_update_split_fused(x, weight, bias, conv_state, query, key, value,
                             key_dim, value_dim, kernel_size,
                             activate=activation == "silu")
    return query, key, value


def _conv_update_split_fused(x, weight, bias, conv_state, query, key, value,
                             key_dim, value_dim, kernel_size, activate):
    """实际发射融合 conv 内核（调用方已完成连续性/形状校验）。

    单独拆出来是为了让参数顺序能在无 GPU 环境下做静态校验
    （见 tests/test_gdn.py 与 tools 校验脚本）。
    """
    batch = x.shape[0]
    conv_dim = 2 * key_dim + value_dim
    state_len = kernel_size - 1
    dtype = x.dtype
    BLOCK_C = 128
    BLOCK_K = triton.next_power_of_2(state_len)
    grid = (triton.cdiv(conv_dim, BLOCK_C), batch)
    runner, args = bound_launch(
        _causal_conv1d_update_split_kernel, "conv_update_split",
        (batch, conv_dim, key_dim, value_dim, kernel_size, state_len,
         BLOCK_C, BLOCK_K, dtype, x.stride(0), conv_state.stride(0),
         query.stride(0), key.stride(0), value.stride(0),
         bias is not None, activate),
        grid,
        x, weight, bias, conv_state, query, key, value,
        x.stride(0), conv_state.stride(0),
        query.stride(0), key.stride(0), value.stride(0),
        state_len,
        constexprs=dict(KEY_DIM=key_dim, VALUE_DIM=value_dim, K=kernel_size,
                        BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K,
                        HAS_BIAS=bias is not None, ACTIVATE=activate),
        num_warps=4,
    )
    runner(*args)


def causal_conv1d_varlen(hidden_states, weight, bias=None, activation="silu", cu_seqlens=None):
    """多序列拼接（flat）张量上的 depthwise 因果卷积。

    先对整条 flat 序列做一次 fused conv，再只修正每个序列开头 K-1 个位置：
    这些位置在 flat conv 里会读到"上一个序列"的 token，需要按边界置零重算。
    修正量只有 S*(K-1) 个位置，代价可忽略，避免逐序列调用卷积。

    性能要点：修正位的写回**不能**使用 data-dependent 的索引（原实现用
    `.nonzero()` 选出有效项，会强制一次 device→host 同步，把已经排队的 GPU
    工作全部等完，是 prefill 阶段 GPU 空转的主要来源）。这里把无效项统一
    "倾倒"到第 T 列（不参与最终输出），所有无效项写入的都是同一列的当前值，
    属于 no-op，因此不会与有效写入竞争，全程无 host 同步。

    Args:
        hidden_states: (B, C, T)，多序列首尾相接
        cu_seqlens: (S+1,) 各序列的起止下标（varlen 约定）
    """
    B, C, T = hidden_states.shape
    K = weight.shape[-1]
    out = F.conv1d(
        hidden_states.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias,
        padding=K - 1, groups=C,
    )                                        # (B, C, T + K - 1)：多出的列恰好给"垃圾桶"留位

    if cu_seqlens is not None and K > 1:
        starts = cu_seqlens[:-1].long()
        ends = cu_seqlens[1:].long()
        # 需要修正的位置：每个序列的前 K-1 个 token
        off = torch.arange(K - 1, device=hidden_states.device)
        pos = starts[:, None] + off[None, :]                 # (S, K-1)
        valid = (pos < ends[:, None]).reshape(-1)            # 段长不足 K-1 时截断
        src = pos.reshape(-1)
        # 无效项（越界或段长不足）→ 垃圾桶列 T；该列不在最终输出里
        tgt = torch.where(valid, src, torch.full_like(src, T))
        acc = out.new_zeros(C, src.numel())
        if bias is not None:
            acc = acc + bias.view(C, 1)          # 重算的位置同样要带上 bias
        seg_start = starts.repeat_interleave(K - 1)
        for j in range(K):
            s = src - (K - 1) + j
            ok = valid & (s >= seg_start)
            # 越界项会被 ok 置零，这里只需保证索引落在 [0, T-1] 内（避免读越界）
            acc = acc + weight[:, j].view(C, 1) * hidden_states[0][:, s.clamp(0, T - 1)] * ok
        cur = out[0][:, tgt]                     # 先读（无效项读到的是垃圾桶列的现值）
        out[0][:, tgt] = torch.where(valid[None, :], acc, cur)

    if activation:
        out = F.silu(out)
    return out[..., :T].to(hidden_states.dtype)


def varlen_conv_state(conv_input, cu_seqlens, pad):
    """取每个序列最后 pad 个输入，拼成 (S, C, pad) 的 conv_state（供 decode 续算）。

    段长不足 pad 时在左侧补零（右对齐），全程无 host 同步。
    """
    B, C, T = conv_input.shape
    starts = cu_seqlens[:-1].long()
    ends = cu_seqlens[1:].long()
    off = torch.arange(pad, device=conv_input.device)
    idx = ends[:, None] - pad + off[None, :]                 # (S, pad)
    valid = idx >= starts[:, None]                           # 越过段首的位置补零
    gathered = conv_input[0][:, idx.clamp(min=0).reshape(-1)].reshape(C, ends.numel(), pad)
    gathered = gathered * valid.unsqueeze(0)
    return gathered.permute(1, 0, 2).contiguous()


# ======================================================================
# FLA recurrent 内核的预绑定启动器（decode 单步专用）
# ======================================================================
# `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` 每次调用要穿过
# autograd.Function、input_guard（对约 20 个入参逐个 .contiguous()）、Heuristics
# 的 lambda 求值，最后才是 Triton binder；实测 ~141us/次，而 decode 每步有 24 个
# GDN 层，单这一项就吃掉 ~3.4ms/step。这里用同一个 kernel、同一组 constexpr 直接
# 预绑定（数值完全等价），只省掉 Python 包装。
_FLA_RECURRENT_CONSTEXPRS = dict(
    USE_G=True, USE_GK=False, USE_GV=False,
    USE_QK_L2NORM_IN_KERNEL=True, IS_BETA_HEADWISE=True,
    USE_INITIAL_STATE=True, STORE_FINAL_STATE=True,
    STATE_V_FIRST=False, IS_VARLEN=False,
    # 门控由 gdn_gate 预先算好（g 为 log 空间 fp32、beta 已过 sigmoid）：
    # 这样喂给内核的是连续张量，绕开了 [b|a] 半区非连续的问题。
    USE_GATE_IN_KERNEL=False, HAS_DT_BIAS=False,
    APPLY_BETA_SIGMOID=False, ALLOW_NEG_EIGVAL=False,
)

_FLA_FALLBACK_WARNED = False


@triton.jit
def _gdn_recurrent_pool_decode_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    state_ptr, slot_ptr,
    layer_stride, slot_stride, scale,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
):
    """单步（T=1）recurrent delta rule，循环状态**直接在池里原地读写**。

    与 FLA 的 `fused_recurrent_gated_delta_rule_fwd_kernel` 逐行等价：同样的
    QK L2 归一化顺序、同样的 `b_h *= exp(g)` 衰减、同样的写入/读出公式，
    只有两处不同：
      1. 初始状态按 (slot_ptr[i_n], layer) 直接寻址，最终状态**原地写回**同一位置
         —— 省掉 gather + scatter 两次 33MB（每层每步）的状态搬运；
      2. 只做 T=1，去掉时间循环与 varlen / 多门控分支（decode 专用）。
    """
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    slot_id = tl.load(slot_ptr + i_n)
    p_state = state_ptr + slot_id * slot_stride + layer_stride + i_hv * K * V

    p_q = q_ptr + (i_n * H + i_h) * K + o_k
    p_k = k_ptr + (i_n * H + i_h) * K + o_k
    p_v = v_ptr + (i_n * HV + i_hv) * V + o_v
    p_g = g_ptr + i_n * HV + i_hv
    p_beta = beta_ptr + i_n * HV + i_hv
    p_o = o_ptr + (i_n * HV + i_hv) * V + o_v

    b_h = tl.load(p_state + o_k[:, None] * V + o_v[None, :],
                  mask=mask_h, other=0.0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale
    b_beta = tl.load(p_beta).to(tl.float32)
    b_g = tl.load(p_g).to(tl.float32)
    b_h *= _fla_exp(b_g)
    b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
    b_h += b_k[:, None] * b_v
    b_o = tl.sum(b_h * b_q[:, None], 0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    tl.store(p_state + o_k[:, None] * V + o_v[None, :],
             b_h.to(p_state.dtype.element_ty), mask=mask_h)


def gdn_recurrent_pool_decode(query, key, value, g, beta, rec_pool, slots, layer_idx):
    """decode 单步的 recurrent delta rule，状态直接在 `rec_pool` 里原地读写。

    Args:
        query/key: (B, 1, num_k_heads, head_k_dim) 连续
        value:     (B, 1, num_v_heads, head_v_dim) 连续
        g:         (B, 1, num_v_heads) float32，log 空间衰减
        beta:      (B, 1, num_v_heads)，已过 sigmoid
        rec_pool:  (num_slots, num_layers, num_v_heads, head_k_dim, head_v_dim) fp32 连续
        slots:     (B,) int32/int64，每个 batch 元素对应的池槽位
        layer_idx: 该 GDN 层在池里的层下标（int）

    返回 (B, 1, num_v_heads, head_v_dim) 的输出；`rec_pool` 对应行被原地更新。
    """
    batch, seq_len, num_k_heads, head_k_dim = key.shape
    num_v_heads, head_v_dim = value.shape[2], value.shape[-1]
    BK = triton.next_power_of_2(head_k_dim)
    BV = min(8, triton.next_power_of_2(head_v_dim))
    scale = head_k_dim ** -0.5
    out = torch.empty(batch, 1, num_v_heads, head_v_dim,
                      dtype=value.dtype, device=value.device)
    grid = (triton.cdiv(head_v_dim, BV), batch * num_v_heads)
    layer_offset = int(layer_idx) * rec_pool.stride(1)   # 该层在池里的绝对偏移
    runner, args = bound_launch(
        _gdn_recurrent_pool_decode_kernel, "gdn_rec_pool",
        (batch, num_k_heads, num_v_heads, head_k_dim, head_v_dim, BK, BV,
         value.dtype, g.dtype, beta.dtype, rec_pool.dtype, slots.dtype,
         rec_pool.stride(0), layer_offset),
        grid,
        query, key, value, g, beta, out,
        rec_pool, slots, layer_offset, rec_pool.stride(0), scale,
        constexprs=dict(H=num_k_heads, HV=num_v_heads, K=head_k_dim, V=head_v_dim,
                        BK=BK, BV=BV),
        num_warps=1,
    )
    runner(*args)
    return out


def fla_recurrent_decode(
    query: torch.Tensor,          # (B, 1, num_k_heads, head_k_dim)
    key: torch.Tensor,
    value: torch.Tensor,          # (B, 1, num_v_heads, head_v_dim)
    g: torch.Tensor,              # (B, 1, num_v_heads) log 空间衰减，float32
    beta: torch.Tensor,           # (B, 1, num_v_heads) 已过 sigmoid
    initial_state: torch.Tensor,  # (B, num_v_heads, head_k_dim, head_v_dim) fp32
    fallback,                     # () -> (out, final_state)，即公开 API 的调用
) -> tuple[torch.Tensor, torch.Tensor]:
    """decode 单步的 recurrent delta rule（预绑定启动器；不可用时回退公开 API）。

    与 `fused_recurrent_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)`
    """
    global _FLA_FALLBACK_WARNED
    if fused_recurrent_gated_delta_rule_fwd_kernel is None:
        return fallback()
    # 内核没有 stride 入参（q/k/v/g/beta/h0 全按连续布局寻址）。预绑定绕开了
    # 公开 API 里 input_guard 的 .contiguous()，所以这里必须自己确认，
    # 非连续时交回公开 API 去拷贝，绝不能带着错误布局发射。
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
            and g.is_contiguous() and beta.is_contiguous()
            and initial_state.is_contiguous()):
        return fallback()
    try:
        batch, seq_len, num_k_heads, head_k_dim = key.shape
        num_v_heads, head_v_dim = value.shape[2], value.shape[-1]
        BK = triton.next_power_of_2(head_k_dim)
        BV = min(8, triton.next_power_of_2(head_v_dim))
        scale = head_k_dim ** -0.5
        out = torch.empty_like(value)
        final_state = query.new_empty(batch, num_v_heads, head_k_dim, head_v_dim,
                                      dtype=torch.float32)
        grid = (triton.cdiv(head_v_dim, BV), batch * num_v_heads)
        runner, args = bound_launch(
            fused_recurrent_gated_delta_rule_fwd_kernel, "fla_recurrent_decode",
            (batch, seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim, BK, BV,
             query.dtype, value.dtype, g.dtype, beta.dtype,
             initial_state.dtype, scale),
            grid,
            query, key, value, g, None, None, beta, None, None,
            out, initial_state, final_state, None, scale, seq_len,
            constexprs=dict(H=num_k_heads, HV=num_v_heads, K=head_k_dim, V=head_v_dim,
                            BK=BK, BV=BV, **_FLA_RECURRENT_CONSTEXPRS),
            num_warps=1,
        )
    except Exception as exc:      # pragma: no cover - 依赖 FLA/Triton 内部实现
        if not _FLA_FALLBACK_WARNED:
            _FLA_FALLBACK_WARNED = True
            warnings.warn(f"FLA 预绑定启动器不可用，回退到公开 API：{exc!r}")
        return fallback()
    runner(*args)
    return out, final_state


# ======================================================================
# GDN 完整层（Qwen3_5GatedDeltaNet）
# ======================================================================


class FusedInputProj(nn.Module):
    """把若干输入投影融合成一个 GEMM（对应 vLLM 的 qkvz / ba 融合投影）。

    checkpoint 里权重是分开的（in_proj_qkv / in_proj_z / in_proj_b / in_proj_a），
    加载时按 shard_id 写入对应分段；前向只发一次 GEMM，减少 decode 的 kernel 数与
    Python/launch 开销。
    """

    def __init__(self, input_size: int, output_sizes: list[int]):
        super().__init__()
        self.output_sizes = output_sizes
        self.weight = nn.Parameter(torch.empty(sum(output_sizes), input_size))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int):
        start = sum(self.output_sizes[:shard_id])
        end = start + self.output_sizes[shard_id]
        param.data[start:end].copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Qwen3.5 GDN 注意力层（与 transformers Qwen3_5GatedDeltaNet 对齐）。

    数据流（与 transformers 实现逐名对应）：

        hidden_states
          ├─ W_qkv → q, k, v（连续布局，key head 在前、value head 在后）
          ├─ W_z   → z（输出门）
          ├─ W_b   → b（写入强度候选）
          └─ W_a   → a（衰减率候选）

        [q,k,v] ── ShortConv（depthwise kernel=K + silu）──► q,k,v 逐 head 重排
        q,k ← L2Norm(q,k)；q ← q·(k_dim)^-0.5
        b → β = σ(b)； a → g = −exp(A_log)·softplus(a + dt_bias)
        q,k ← repeat_interleave( num_v_heads/num_k_heads )   # GQA 式扩展
        y  ← ΔRule(q, k, v, g, β)       # recurrent(chunk) 内核
        o  ← RMSNormGated(y, z)          # RMS 归一化 × silu(z)
        out ← W_out(o)

    注意与 Qwen3-Next 的差异（本项目按 Qwen3.5 的独立投影实现）：
        - checkpoint 权重拆成 in_proj_qkv / in_proj_z / in_proj_b / in_proj_a
          （Qwen3-Next 是融合的 in_proj_qkvz / in_proj_ba）
        - q/k/v 为连续排列（vLLM 加载时用 gqa_interleaved_layout=False）

    Args:
        hidden_size:            输入/输出维
        linear_num_value_heads: value head 数（默认 32，越大记忆容量越大）
        linear_num_key_heads:   key head 数（默认 16，GQA 分组粒度）
        linear_key_head_dim:    key 头维（默认 128）
        linear_value_head_dim:  value 头维（默认 128）
        linear_conv_kernel_dim: short conv 核长（默认 4）
        rms_norm_eps:           RMSNormGated 的 eps
    """

    def __init__(
        self,
        hidden_size: int,
        linear_num_value_heads: int = 32,
        linear_num_key_heads: int = 16,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = linear_num_value_heads
        self.num_k_heads = linear_num_key_heads
        self.head_k_dim = linear_key_head_dim
        self.head_v_dim = linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = linear_conv_kernel_dim
        self.layer_idx = layer_idx

        # ---- 投影（Qwen3.5 风格：checkpoint 是 4 个独立投影，这里融合成 2 个 GEMM）----
        self.in_proj_qkvz = FusedInputProj(
            hidden_size, [self.key_dim * 2 + self.value_dim, self.value_dim])
        self.in_proj_ba = FusedInputProj(
            hidden_size, [self.num_v_heads, self.num_v_heads])

        # ---- short conv：对 [q;k;v] 拼接的 depthwise 因果卷积 ----
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,                    # 每通道独立卷积（depthwise）
            padding=self.conv_kernel_size - 1,       # 因果 padding（结果裁回）
        )

        # ---- 时间离散化参数（决定衰减率 α）----
        # dt_bias 初始为 1（保持 softplus 输入为正）
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        # A_log：每个 value head 一个对数标度，初始采样 ~U(0.01, 16) 后取 log；
        # α = exp[−exp(A)·softplus(a + dt_bias)] 恒在 (0,1)
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # ---- 输出门：RMS 归一化 × silu(z) ----
        self.norm = RMSNormGated(self.head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,                                          # (B, L, D)
        conv_state: torch.Tensor | None = None,                               # (B, C, K-1)
        recurrent_state: torch.Tensor | None = None,                          # (B, V, Kd, Vd)
        cu_seqlens: torch.Tensor | None = None,                               # (S+1,) varlen 拼接边界
        rec_pool: torch.Tensor | None = None,                                 # (S, L, V, Kd, Vd)
        rec_slots: torch.Tensor | None = None,                                # (B,) 池槽位
        rec_layer: int | None = None,                                         # 本层在池里的层下标
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向：返回 (输出, 新的 conv_state, 新的 recurrent_state)。

        三种模式：
        - varlen prefill（cu_seqlens 非空）：多序列首尾相接，conv 与 chunk 内核各一次算完
        - decode（L == 1 且带 conv_state）：融合 conv 增量内核（状态移位 + 卷积 +
          激活 + 按 q/k/v 分段写出）+ 预绑定启动器的 recurrent 内核
        - 普通 prefill（L > 1）：conv 全量 + chunk 内核

        decode 时若给了 rec_pool/rec_slots/rec_layer，recurrent 那一步会走
        「池内原地读写」的融合内核（省掉 gather + scatter），此时返回的
        new_recurrent_state 为 None（状态已在池里更新完毕）。
        """
        batch, seq_len, _ = hidden_states.shape
        varlen = cu_seqlens is not None
        # 单步 decode：conv/recurrent 增量路径（每序列 1 个 token）
        is_step_decode = (not varlen) and conv_state is not None and seq_len == 1

        # 1) 投影（qkvz / ba 各一次 GEMM，split 都是 view，不产生额外 kernel）
        qkvz = self.in_proj_qkvz(hidden_states)
        qkv, z = torch.split(
            qkvz, [self.key_dim * 2 + self.value_dim, self.value_dim], dim=-1)
        mixed_qkv = qkv.transpose(1, 2)                               # (B, C, L) 给 conv
        z = z.reshape(batch, seq_len, -1, self.head_v_dim)
        ba = self.in_proj_ba(hidden_states)
        # 门控参数：β ∈ (0,1)；g ∈ (−∞, 0]（log 空间）。
        # 融合成一个 Triton kernel（对应 vLLM 的 fused gating），替代原先
        # sigmoid / cast / add-bias / softplus / exp(A_log) / mul / neg 等 ~8 个
        # 逐元素 kernel；CPU 或非连续输入走参考实现。
        #
        # 为什么 decode 也用这个 kernel 而不是让 FLA 内核内部算门控：
        # 内核内的门控要求把 ba 的 [b|a] 两个半区分别喂进去，而它们是行 stride = 2H
        # 的**非连续视图**（FLA 内核没有 stride 入参）—— 公开 API 靠 input_guard 的
        # .contiguous() 掩盖了这一点，预绑定路径没有那层包装，会静默读错 batch 维。
        # gdn_gate 产出连续的 beta / g，既正确又只多一次很便宜的小 kernel。
        if ba.is_cuda and ba.stride(-1) == 1 and ba.dtype in (torch.bfloat16, torch.float16):
            beta, g = gdn_gate(ba, self.A_log, self.dt_bias)
        else:
            beta, g = gdn_gate_native(ba, self.A_log, self.dt_bias)

        # 2) short conv（varlen 全量 / decode 增量 / 普通 prefill 全量）
        conv_weight = self.conv1d.weight.squeeze(1)
        if varlen:
            conv_input = mixed_qkv
            mixed_out = causal_conv1d_varlen(
                conv_input, conv_weight, self.conv1d.bias, activation="silu",
                cu_seqlens=cu_seqlens)
            new_conv_state = varlen_conv_state(
                conv_input, cu_seqlens, self.conv_kernel_size - 1)
            query, key, value = torch.split(
                mixed_out.transpose(1, 2),
                [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        elif is_step_decode:
            # decode 单步：一次融合 kernel 完成"状态移位 + 卷积 + 激活"，并把
            # [q|k|v] 三段分别写成连续缓冲区，省掉 split 后 q/k/v 各一次 contiguous 拷贝
            query, key, value = causal_conv1d_update_split(
                mixed_qkv, conv_state, conv_weight, self.conv1d.bias,
                self.key_dim, "silu")
            new_conv_state = conv_state
        else:
            # prefill 全量：一次算完，并导出最近 K-1 个输入作为后续 decode 的起点
            mixed_out, new_conv_state = causal_conv1d_fn(
                mixed_qkv, conv_weight, self.conv1d.bias,
                activation="silu", return_state=True)
            query, key, value = torch.split(
                mixed_out.transpose(1, 2),
                [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        # 3) 拆成 head 维度（decode 路径进来的三块本身连续，reshape 是零拷贝视图）
        query = query.reshape(batch, seq_len, -1, self.head_k_dim)   # (B, L, nk, Kd)
        key = key.reshape(batch, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch, seq_len, -1, self.head_v_dim)   # (B, L, nv, Vd)

        # 5) GQA 交给 FLA 的 GVA（HV > H 时内核内部按 key head 分组），与显式
        #    repeat_interleave 数值完全等价，省掉一次 q/k 复制
        # 6) delta rule 内核（均为 FLA Triton 实现；scale 与 L2 归一化都在内核内完成）
        if varlen:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, cu_seqlens=cu_seqlens,
                initial_state=recurrent_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True)
        elif is_step_decode:
            # decode：走预绑定启动器（同一个 kernel / constexpr，省掉 FLA 的
            # autograd.Function + input_guard + Heuristics + binder 开销）
            if rec_pool is not None:
                # 融合路径：状态直接在池里原地读写（省掉每层每步 2×33MB 的搬运）
                try:
                    core_attn_out = gdn_recurrent_pool_decode(
                        query, key, value, g, beta, rec_pool, rec_slots, rec_layer)
                    last_recurrent_state = None
                except Exception as exc:   # pragma: no cover - 兜底
                    if not _FLA_FALLBACK_WARNED:
                        _FLA_FALLBACK_WARNED = True
                        warnings.warn(f"池内 recurrent 内核不可用，回退 gather+FLA：{exc!r}")
                    rec_state = rec_pool[rec_slots.long(), rec_layer]
                    core_attn_out, last_recurrent_state = fla_recurrent_decode(
                        query, key, value, g, beta, rec_state,
                        fallback=lambda: fused_recurrent_gated_delta_rule(
                            query, key, value, g=g, beta=beta,
                            initial_state=rec_state, output_final_state=True,
                            use_qk_l2norm_in_kernel=True))
            else:
                core_attn_out, last_recurrent_state = fla_recurrent_decode(
                    query, key, value, g, beta, recurrent_state,
                    fallback=lambda: fused_recurrent_gated_delta_rule(
                        query, key, value, g=g, beta=beta,
                        initial_state=recurrent_state, output_final_state=True,
                        use_qk_l2norm_in_kernel=True))
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True)

        # 7) 输出门 + 投影（与 transformers 一致：展平到 head_v_dim 再逐个调制）
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm(core_attn_out, z)                       # RMS(o) × silu(z)
        out = self.out_proj(out.reshape(batch, seq_len, self.value_dim))
        return out, new_conv_state, last_recurrent_state
