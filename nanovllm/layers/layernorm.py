import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.layers.triton_launch import block_n, bound_launch


@triton.jit
def _rms_norm_gated_kernel(
    X, Y, W, Z, Rstd,
    stride_x_row, stride_y_row, stride_z_row,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    """融合 RMSNorm + silu(gate)（对齐 vLLM 的 layernorm_gated Triton 内核）。

    一次 kernel 完成：rms 归一化 → weight 缩放 → silu(z) 调制，
    省掉中间张量的多次读写与 kernel 往返。
    """
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    Z += row * stride_z_row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(tl.where(mask, x, 0.0) * tl.where(mask, x, 0.0), axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    z = tl.load(Z + cols, mask=mask).to(tl.float32)
    y = x * rstd * w * (z * tl.sigmoid(z))
    tl.store(Y + cols, y, mask=mask)


def rms_norm_gated(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float):
    """走融合 Triton 内核的 RMSNormGated（要求 CUDA + 最后一维连续）。"""
    x2 = x.reshape(-1, x.shape[-1])
    gate2 = gate.reshape(-1, gate.shape[-1])
    out = torch.empty_like(x2)
    M, N = x2.shape
    BLOCK_N = block_n(N, x2.element_size())
    rstd = torch.empty(M, dtype=torch.float32, device=x2.device)
    runner, args = bound_launch(
        _rms_norm_gated_kernel, "gated",
        (M, N, x2.dtype, gate2.dtype, weight.dtype,
         x2.stride(0), out.stride(0), gate2.stride(0), BLOCK_N),
        (M,), x2, out, weight, gate2, rstd,
        x2.stride(0), out.stride(0), gate2.stride(0), M, N, eps,
        constexprs=dict(BLOCK_N=BLOCK_N),
        num_warps=min(max(BLOCK_N // 256, 1), 8),
    )
    runner(*args)
    return out.reshape(x.shape)


@triton.jit
def _rms_norm_kernel(
    X, Y, W,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_N: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
):
    """融合 RMSNorm：rms 归一化 + 权重缩放，一次 kernel 完成一行。

    ZERO_CENTERED=True 时按 Qwen3.5 的 zero-centered 约定用 (1 + w)。
    浮点计算全在 float32，写出时按 Y 的 dtype 截断（与参考实现一致）。
    """
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        w = w + 1.0
    tl.store(Y + cols, x * rstd * w, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered: bool = False):
    """融合 RMSNorm（替换逐算子的 eager/torch.compile 实现）。

    只发一次 kernel，省掉 pow / mean / rsqrt / mul / cast 等 5~8 次算子派发
    与 dtype 往返（Qwen3.5 每步有 ~65 次 norm 调用，这里是纯 CPU 开销）。
    """
    x2 = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(x2)
    M, N = x2.shape
    BLOCK_N = block_n(N, x2.element_size())
    num_warps = min(max(BLOCK_N // 512, 1), 8)
    runner, args = bound_launch(
        _rms_norm_kernel, "rms",
        (M, N, x2.dtype, weight.dtype, x2.stride(0), out.stride(0),
         BLOCK_N, zero_centered, num_warps),
        (M,), x2, out, weight, x2.stride(0), out.stride(0), N, eps,
        constexprs=dict(BLOCK_N=BLOCK_N, ZERO_CENTERED=zero_centered),
        num_warps=num_warps,
    )
    runner(*args)
    return out.reshape(x.shape)


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class ZeroCenteredRMSNorm(nn.Module):
    """Zero-centered RMSNorm（Qwen3.5 / Qwen3.8-27B 使用的归一化变体）。

    与普通 RMSNorm 的唯一区别在权重初始值与用法：
    - 普通 RMSNorm:  weight = ones，输出 = rms(x) * weight
    - Zero-centered: weight = zeros，输出 = rms(x) * (1 + weight)

    即以 1 为中心做归一化（"zero-centered" 指的是权重围绕 1 学习，
    初始化时退化为标准的 RMS 归一化，训练中通过 (1 + w) 微调缩放）。
    这一设计约束了 LN weight 的数值范围，提升了深度网络的训练稳定性
    （Qwen3.8 论文中所有 RMSNorm（含 q/k norm、decoder norm）均统一使用该形式）。

    注：权重直接加载自 checkpoint（存储的就是 w），前向时不需要额外变换。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 与普通 RMSNorm 不同：weight 初始化为 0（而非 1）
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 逐元素平方求均值 → rsqrt 得 RMS 倒数 → 缩放
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        # 全程 float32 计算以保证精度（RMS 归一化本身对低精度敏感）
        output = self._norm(x.float())
        # (1 + weight) 形式：初始时 weight=0 即标准 RMSNorm
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CUDA + 末维连续时走融合 Triton 内核（一次 kernel 完成归一化与缩放），
        # 避免 torch.compile 的 dynamo guard/FxGraph 调用开销（每步约 65 次）。
        if x.is_cuda and x.stride(-1) == 1:
            return rms_norm(x, self.weight, self.eps, zero_centered=True)
        return self._forward_native(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class RMSNormGated(nn.Module):
    """Gated RMSNorm（GDN 线性注意力层的输出门）。

    数学形式（Qwen3.5 GDN 层，transformers Qwen3_5RMSNormGated）：

        x_norm = RMSNorm(x)                      # 先归一化
        out    = x_norm * silu(gate)             # 再用输入相关的 gate 逐元素调制

    对应论文式(11) `o_t = W_o[σ(W_z x_t) ⊙ RMSNorm(y_t)]` 中的归一化+调制部分
    （Qwen3.5 用 silu 作为 gate 激活，Flash-Next 才改为 sigmoid）。

    作用：GDN 循环输出 y 的数值范围无界，先 RMS 归一化保证稳定，
    再通过 gate 让模型按 token 内容自主学习"这一层要放大多大输出"。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 普通 RMSNorm 权重（ones）
        self.variance_epsilon = eps
        self.activation = "silu"  # gate 激活函数（Qwen3.5 版本固定为 silu）

    def _forward_native(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # 归一化与 gate 均在 float32 下计算，避免低精度累积误差
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        # gate 形状须与 hidden_states 尾部维度对齐（都是 head_v_dim）→ 逐元素相乘
        hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # CUDA 且末维连续时走融合 Triton 内核（对齐 vLLM 的 layernorm_gated）：
        # 一次算完 rms 归一化 + weight 缩放 + silu(gate) 调制；
        # 其它情况（CPU、非连续）走 reference 实现，保证可移植与测试可跑。
        if (hidden_states.is_cuda and hidden_states.stride(-1) == 1
                and gate.stride(-1) == 1 and hidden_states.shape == gate.shape):
            return rms_norm_gated(hidden_states, self.weight, gate, self.variance_epsilon)
        return self._forward_native(hidden_states, gate)
