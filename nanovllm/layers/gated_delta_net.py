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

本文件提供两种计算模式（数学等价，不同场景用）：
    - torch_recurrent_gated_delta_rule：逐 token 循环，用于 decode（单步）
    - torch_chunk_gated_delta_rule  ：按 chunk 并行，用于 prefill（长序列）
    - Qwen3_5GatedDeltaNet          ：完整层（投影 + short conv + 门控 + 输出）

参考实现：transformers modeling_qwen3_5.py（Apache 2.0）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.layernorm import RMSNormGated


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
        conv_state = hidden_states[:, :, -padding:]
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


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """沿最后一维做 L2 归一化（与 FLA 库保持一致）。

    QB2019：q/k 归一化后，delta rule 的 rank-one 更新范数有界，
    避免状态 S 数值爆炸；同时统一了 q·k 的内积尺度。
    """
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# ======================================================================
# Gated Delta Rule 内核之（1）：逐 token 递推（decode 用）
# ======================================================================
def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """逐 token 循环的 gated delta rule。

    形状约定（与 FLA/transformers 一致）：
        query/key: (batch, seq, num_k_heads, k_dim)      （key head 数）
        value    : (batch, seq, num_v_heads, v_dim)      （value head 数）
        g, beta  : (batch, seq, num_v_heads)
        state    : (batch, num_v_heads, k_dim, v_dim)

    用 gqa_interleave 后 query/key 已重复扩展为 num_v_heads 份，
    所以这里逐 head 独立递推。

    该实现是"语义教科书"：直接按 Eq.1-5 逐步执行，便于理解原理；
    每步开销是 (k_dim × v_dim) 的小矩阵运算，decode（seq=1）时刚好。

    Args:
        g:     log 空间衰减（≤0），state 每步乘 exp(g)
        beta:  写入强度（0~1）
        use_qk_l2norm_in_kernel: 是否在循环内对 q/k 做 L2 归一化
    """
    initial_dtype = query.dtype
    batch, seq_len, _, k_dim = key.shape
    num_v_heads, v_dim = value.shape[-2:]
    decay = g  # 参数名与 flash_linear_attention 一致

    # 统一转成 fp32（delta 递推对数值精度敏感）+ 轴序 [B, H, L, D]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # query 按 head 维开根做缩放（与 FLA 对齐，保证注意力量级稳定）
    query = query / (query.shape[-1] ** 0.5)

    if initial_state is None:
        recurrent_state = torch.zeros(
            (batch, num_v_heads, k_dim, v_dim), dtype=value.dtype, device=value.device)
    else:
        recurrent_state = initial_state.to(value)

    output = torch.zeros_like(value)
    # ----- 核心循环：严格按 Eq.1-5 -----
    for i in range(seq_len):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # (1) 衰减旧状态：S̃ = S · α
        decay_t = decay[:, :, i].exp()[..., None, None]
        recurrent_state = recurrent_state * decay_t
        # (2)+(3) delta 写入：S ← S̃ + β·k⊗(v − S̃ᵀk)
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)  # S̃ᵀk → 旧记忆的预测
        delta = (v_t - kv_mem) * beta_t                             # 误差 × 写入强度
        recurrent_state = recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # (4) 读出：y = Sᵀq
        output[:, :, i] = (recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        recurrent_state = None
    return output.transpose(1, 2).contiguous().to(initial_dtype), recurrent_state


# ======================================================================
# Gated Delta Rule 内核之（2）：chunk 并行（prefill 用）
# ======================================================================
def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """chunk 并行版 gated delta rule（与 recurrent 版本数学等价，用于 prefill）。

    动机：recurrent 版逐 token 串行，GPU 利用率差。Δrule 更新可以写成
    一个"线性 sketch"（UT 上三角变换系统），使得：
        - chunk 内部的全部矩阵运算（intra-chunk attn、UT 求解）互不依赖 → 并行
        - 只有跨 chunk 的状态传递是顺序的（每 chunk 一次小矩阵乘）

    具体分工（transformers 实现）：
       Phase 1（并行 part）：对每个 chunk 预计算
           ut_system      = (k_β ⊗ k) * pair_decay    （chunk 内 delta 累积的系数矩阵）
           intra_chunk    = (q ⊗ k) * pair_decay      （chunk 内线性注意力项）
           k_cumdecay     = UT⁻¹(decayed_k_β)         （消去"旧状态预测"）
           new_values     = UT⁻¹(v_β)                 （UT 上三角求解）
       Phase 2（顺序 part）：跨 chunk 扫描
           每个 chunk：y += intra + q @ state
                       state = state * chunk_decay + k̄ᵀ @ v_new
    整个算法的计算量从 O(n²) 降到 chunk 粒度的并行（~n/chunk_size 步顺序扫描）。

    Args 与 recurrent 版相同，额外：
        chunk_size: 序列切块大小（Qwen3.5 默认 64）
    """
    initial_dtype = query.dtype
    batch, seq_len, _, k_dim = key.shape
    num_v_heads, v_dim = value.shape[-2:]
    recurrent_state_shape = (batch, num_v_heads, k_dim, v_dim)
    padded_output_shape = (batch, num_v_heads, -1, v_dim)   # -1 由 pad 后长度决定
    decay = g

    # 统一 fp32 + 轴序 [B, H, L, D]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query * (query.shape[-1] ** -0.5)

    # 序列长度补齐到 chunk_size 的整数倍（右侧补 0，alignment 对齐核心理念）
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    total_seq_len = seq_len + pad_size
    num_chunks = total_seq_len // chunk_size

    # β 作用到 k/v 上（"学习率"缩放），再切块 → [B, H, n_chunk, chunk, D]
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    # chunk 内因果掩码（上三角屏蔽：chunk 内不能看到自己之后的位置）
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                     device=query.device).triu(1)
    # 对数空间累积衰减：cum_decay[..., t] = Σ_{j≤t} g_j
    cum_decay = decay.cumsum(dim=3)

    # ---- Phase 1：chunk 内并行部分 ----
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()          # 位置 j→i 之间累积的衰减系数

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # 上三角（unit lower）线性系统求解：把 chunk 内 k 步 delta 更新
    # 压缩成一次矩阵运算（delta rule 的线性可加性）
    new_values = torch.linalg.solve_triangular(
        ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(
        ut_system, decayed_k_beta, upper=False, unitriangular=True)

    if initial_state is None:
        last_recurrent_state = torch.zeros(
            recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    # 衰减因子拆到每 chunk 头尾，让扫描里只需一次标量-矩阵乘
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # ---- Phase 2：跨 chunk 顺序扫描（每 chunk 一次状态更新） ----
    for i in range(num_chunks):
        # 本 chunk 对旧状态的"修正"：新值中减去旧状态已能预测的部分（delta rule 本质）
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state   # 读旧记忆
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        # 状态推进：S ← S·α_chunk + k̄ᵀ·v_new
        last_recurrent_state = (
            last_recurrent_state * chunk_decay[:, :, i]
            + key[:, :, i].transpose(-1, -2) @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(padded_output_shape)[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


# ======================================================================
# GDN 完整层（Qwen3_5GatedDeltaNet）
# ======================================================================
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

        # ---- 投影（Qwen3.5 风格：4 个独立投影，bias=False）----
        self.in_proj_qkv = nn.Linear(hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, self.num_v_heads, bias=False)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向：返回 (输出, 新的 conv_state, 新的 recurrent_state)。

        - prefill（L > 1）：causal_conv1d_fn 一次算完，recurrent_state 为 None 时用 chunk 内核
        - decode（L == 1）：causal_conv1d_update 增量算，recurrent_state 必传 → recurrent 内核
        """
        batch, seq_len, _ = hidden_states.shape

        # 1) 投影
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)   # (B, C, L) 给 conv
        z = self.in_proj_z(hidden_states).reshape(batch, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # 2) short conv（prefill 全量 / decode 增量）
        new_conv_state = conv_state
        if new_conv_state is not None and seq_len == 1:
            # decode 单步：拼上历史 → 一次无 padding 卷积，同时原地更新 state
            mixed_qkv = causal_conv1d_update(
                mixed_qkv, new_conv_state,
                self.conv1d.weight.squeeze(1), self.conv1d.bias, "silu")
        else:
            # prefill 全量：一次算完，并导出最近 K-1 个输入作为后续 decode 的起点
            mixed_qkv, new_conv_state = causal_conv1d_fn(
                mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                activation="silu", return_state=True)
        mixed_qkv = mixed_qkv.transpose(1, 2)                             # 回到 (B, L, C)

        # 3) 拆分 q/k/v（连续布局）并做 head 重排
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, seq_len, -1, self.head_k_dim)   # (B, L, nk, Kd)
        key = key.reshape(batch, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch, seq_len, -1, self.head_v_dim)   # (B, L, nv, Vd)

        # 4) 门控参数：β ∈ (0,1)；g ∈ (−∞, 0]（log 空间）
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # 5) GQA 扩展：每个 key head 复制 到 num_v_heads / num_k_heads 份
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # 6) delta rule 内核（decode=recurrent / prefill=chunk）
        if conv_state is not None and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True)
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True)

        # 7) 输出门 + 投影（与 transformers 一致：展平到 head_v_dim 再逐个调制）
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm(core_attn_out, z)                       # RMS(o) × silu(z)
        out = self.out_proj(out.reshape(batch, seq_len, self.value_dim))
        # 解码路径的 conv_state 已被原地更新（causal_conv1d_update），直接回传
        return out, new_conv_state, last_recurrent_state