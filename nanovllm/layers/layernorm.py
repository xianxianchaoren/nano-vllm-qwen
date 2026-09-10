import torch
from torch import nn


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

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全程 float32 计算以保证精度（RMS 归一化本身对低精度敏感）
        output = self._norm(x.float())
        # (1 + weight) 形式：初始时 weight=0 即标准 RMSNorm
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

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

    @torch.compile
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # 归一化与 gate 均在 float32 下计算，避免低精度累积误差
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        # gate 形状须与 hidden_states 尾部维度对齐（都是 head_v_dim）→ 逐元素相乘
        hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
