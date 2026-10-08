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

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from nanovllm.layers.layernorm import RMSNormGated


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

    beta 为输入 dtype，g 为 float32。
    """
    *lead, two_h = ba.shape
    H = two_h // 2
    ba2 = ba.reshape(-1, two_h)
    total = ba2.shape[0] * H
    beta = torch.empty((ba2.shape[0], H), dtype=ba.dtype, device=ba.device)
    g = torch.empty((ba2.shape[0], H), dtype=torch.float32, device=ba.device)
    BLOCK = 256
    _gdn_gate_kernel[(triton.cdiv(total, BLOCK),)](
        ba2, a_log, dt_bias, g, beta, total, H=H, BLOCK=BLOCK, num_warps=4)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向：返回 (输出, 新的 conv_state, 新的 recurrent_state)。

        三种模式：
        - varlen prefill（cu_seqlens 非空）：多序列首尾相接，conv 与 chunk 内核各一次算完
        - decode（L == 1 且带 conv_state）：conv 增量 + recurrent 内核
        - 普通 prefill（L > 1）：conv 全量 + chunk 内核
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
        # decode 单步直接把原始 ba 交给 FLA 的 recurrent 内核，由内核内部完成
        # sigmoid(b) 与 -exp(A)·softplus(a + dt_bias)（对应 vLLM 融合门控的 decode
        # 路径），省掉一次独立 kernel 及其张量分配；chunk 内核不支持融合门控，
        # prefill 仍用下面的融合 Triton kernel 预先算好（替代原先 ~8 个逐元素 kernel）。
        if is_step_decode:
            b_raw, a_raw = ba.split(self.num_v_heads, dim=-1)
            beta = g = None
        elif ba.is_cuda and ba.stride(-1) == 1 and ba.dtype in (torch.bfloat16, torch.float16):
            beta, g = gdn_gate(ba, self.A_log, self.dt_bias)
        else:
            beta, g = gdn_gate_native(ba, self.A_log, self.dt_bias)

        # 2) short conv（varlen 全量 / decode 增量 / 普通 prefill 全量）
        conv_weight = self.conv1d.weight.squeeze(1)
        if varlen:
            conv_input = mixed_qkv
            mixed_qkv = causal_conv1d_varlen(
                conv_input, conv_weight, self.conv1d.bias, activation="silu",
                cu_seqlens=cu_seqlens)
            new_conv_state = varlen_conv_state(
                conv_input, cu_seqlens, self.conv_kernel_size - 1)
        elif conv_state is not None and seq_len == 1:
            # decode 单步：拼上历史 → 一次无 padding 卷积，同时原地更新 state
            mixed_qkv = causal_conv1d_update(
                mixed_qkv, conv_state, conv_weight, self.conv1d.bias, "silu")
            new_conv_state = conv_state
        else:
            # prefill 全量：一次算完，并导出最近 K-1 个输入作为后续 decode 的起点
            mixed_qkv, new_conv_state = causal_conv1d_fn(
                mixed_qkv, conv_weight, self.conv1d.bias,
                activation="silu", return_state=True)
        mixed_qkv = mixed_qkv.transpose(1, 2)                             # 回到 (B, L, C)

        # 3) 拆分 q/k/v（连续布局）
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
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
        elif conv_state is not None and seq_len == 1:
            core_attn_out, last_recurrent_state = fused_recurrent_gated_delta_rule(
                query, key, value, g=a_raw, beta=b_raw,
                A_log=self.A_log, dt_bias=self.dt_bias,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                initial_state=recurrent_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True)
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
