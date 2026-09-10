from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """应用旋转位置编码（RoPE）。

    排列方式为 NeoX/GLM 的 chunk 风格（rotate-half，非 interleaved）：
    把向量沿最后一维对半切开 x = [x1, x2]，旋转为
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
    与 transformers 中 Qwen3.5 的 `apply_rotary_pos_emb`（GLM 风格）数学等价。

    partial rotary（rotary_dim < head_dim）时，只旋转前 rotary_dim 维，
    剩余维度原样透传（Qwen3.5/Qwen3.8 full attention 用 partial
    rotary factor，例如 0.25 表示只有 1/4 的维度携带位置信息）。

    Args:
        x: [..., head_dim]
        cos/sin: [..., rotary_dim // 2]（每个频率一个 cos/sin，与半切后的 x 对齐）
        rotary_dim: 参与旋转的维度数；None 表示全维度旋转
    """
    if rotary_dim is None or rotary_dim == x.shape[-1]:
        # 全维度旋转：直接对半切分
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)
    # partial rotary：只旋转前 rotary_dim 维，其余 pass-through
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = torch.chunk(x_rot.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2, x_pass), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """预计算 cos/sin 缓存的 RoPE 模块。

    频率与 transformers 的 `compute_default_rope_parameters` 一致：
        inv_freq = base ** (-arange(0, rotary_dim, 2) / rotary_dim)
        freqs    = positions ⊗ inv_freq
        cache    = [cos(freqs); sin(freqs)]          # 前一半 cos、后一半 sin
    索引缓存比每次实时计算快（decode 阶段每步重查 positions 即可）。

    partial rotary（Qwen3.5 full attention / QSA indexer 需要）：
        rotary_dim 可以小于 head_size，仅前 rotary_dim 维参与旋转。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int | None = None,
        max_position_embeddings: int = 4096 * 32,
        base: float = 10000,
        partial_rotary_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # rotary_dim 未显式给出时由 partial_rotary_factor 决定；
        # 兼容旧调用方式（原实现强制 rotary_dim == head_size）
        self.rotary_dim = rotary_dim if rotary_dim is not None \
            else int(head_size * partial_rotary_factor)
        # 逐对频率生成（rotary_dim // 2 个）
        inv_freq = 1.0 / (base**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # cache 形状 [max_position_embeddings, rotary_dim]：
        #   前半列 = cos、后半列 = sin，forward 时按列切开
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin, self.rotary_dim)
        key = apply_rotary_emb(key, cos, sin, self.rotary_dim)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int | None = None,
    max_position: int = 4096 * 32,
    base: float = 10000,
    partial_rotary_factor: float = 1.0,
):
    """按参数获取（缓存的）RoPE 模块。

    现有调用（qwen3.py 中 `get_rope(head_size, rotary_dim=head_size, ...)`）
    传了 rotary_dim 且等于 head_size，走全旋转路径，行为不变。
    """
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        partial_rotary_factor,
    )
    return rotary_emb