"""Qwen3.5 文本模型（Qwen3.8-27B 的架构 = qwen3_5_text）。

架构速览（与 transformers modeling_qwen3_5.py / vLLM qwen3_5.py 对齐）：
- 每层按 config.layer_types 分派：`linear_attention`（GDN）或 `full_attention`
- 64 层结构（27B：16 full + 48 linear，full_attention_interval=4）
- 所有 Norm 使用 Zero-Centered RMSNorm（(1+w) 形式）
- full attention 层：gated QKV（query 输出 2 倍宽，一半 gate）+ q/k norm + partial RoPE
- GDN 层：Qwen3_5GatedDeltaNet（无 RoPE，固定循环状态）

关于文件命名：模型家族名是 "Qwen3.8"，但架构代号是 "qwen3_5_text"，
vLLM/transformers 均按架构代号组织文件（qwen3_5.py），此处保持一致。

forward 语义（与 nano-vllm 现有 qwen3.py 一致）：
- 输入为 **flat 拼接**的 token（多序列 varlen 表示）
- cu_seqlens / seq_slots 从全局 context 读取（由 ModelRunner 在 run 前 set_context）
  - cu_seqlens: 各序列在 flat 序列中的边界（fla/flash-attn varlen 约定）
  - seq_slots : 每个 token 所属序列在"状态池"中的槽位（GDN 寻址用）
- GDN 状态（卷积状态 + 循环状态）存放在模型持有的状态池中，随槽位读写
"""
import torch
from torch import nn

from nanovllm.layers.layernorm import ZeroCenteredRMSNorm
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.gated_delta_net import Qwen3_5GatedDeltaNet
from nanovllm.utils.context import get_context


def _num_linear_layers(config) -> int:
    """统计 layer_types 中 linear_attention 的层数（决定状态池数量）。"""
    return sum(1 for t in config.layer_types if t == "linear_attention")


class Qwen3_5Attention(nn.Module):
    """Full attention 层（Qwen3.5 的 gated 多头注意力）。

    与普通 MHA 的差异：
    1. qkv_proj 的 Q 部分输出 = 2 × head 数 × head_dim（一半 query、一半 gate）；
       注意力输出逐位置 × sigmoid(gate)，再进 o_proj —— 类似 MLA 的 output gate。
    2. q/k 各自过一个 Zero-Centered RMSNorm（作用于 head_dim）。
    3. RoPE 为 partial rotary（仅前 partial_rotary_factor 比例维度旋转）。
    4. 标准 GQA + PagedKV 注意力（复用现有 Attention 层与 KV cache 机制）。
    """

    def __init__(self, config, layer_idx=0) -> None:
        super().__init__()
        tp_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        # Q 头数 ×2：一半用于 query，一半用于 output gate
        self.total_num_heads = config.num_attention_heads * 2
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.q_size = self.num_heads // 2 * self.head_dim      # 实际 query 维
        self.num_q_heads = self.num_heads // 2                 # 实际 query head 数（另一半是 output gate）
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=bias)
        self.o_proj = RowParallelLinear(self.total_num_heads // 2 * self.head_dim,
                                        self.hidden_size, bias=False)
        # QK Norm：zero-centered，作用于 head_dim
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Partial RoPE。
        # transformers 5.x 起 rope_theta 不再作为顶层属性挂在 text config 上，
        # 统一从 rope_parameters 读取（partial_rotary_factor 仍保留旧属性）。
        rope_params = getattr(config, "rope_parameters", None) or {}
        partial = getattr(config, "partial_rotary_factor", None)
        if partial is None:
            partial = rope_params.get("partial_rotary_factor", 1.0)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = rope_params.get("rope_theta", 10000.0)
        max_pos = getattr(config, "max_position_embeddings", 262144) or 262144
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=None,
                                   max_position=max_pos, base=rope_theta,
                                   partial_rotary_factor=partial)
        # 延迟导入：Attention 依赖 flash_attn/triton（GPU 环境才可用），
        # 避免构造 GDN-only 模型时引入重依赖
        from nanovllm.layers.attention import Attention
        self.attn = Attention(self.num_q_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        # 拆出 [gated_query, key, value]：query 槽位宽度是 2×q_size
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        # 必须在 reshape 成 (N, heads, head_dim) 之后再调用 Attention / RoPE：
        # 二者都按最后一维（head_dim）计算，扁平张量会被错误地对半切分。
        q_gate = q_gate.view(-1, self.num_q_heads, self.head_dim * 2)
        q, gate = q_gate.chunk(2, dim=-1)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # QK Norm（zero-centered，作用于 head_dim）
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Partial RoPE（作用于 (N, heads, head_dim)）
        q, k = self.rotary_emb(positions, q, k)
        # 注意力（q: (N, nq, hd)，k/v: (N, nkv, hd)）
        o = self.attn(q, k, v)
        if o.dim() == 4:                     # decode: flash-attn 返回 (B, 1, nq, hd)
            o = o.squeeze(1)
        o = o * torch.sigmoid(gate)          # output gate
        return self.o_proj(o.flatten(1, -1))


class Qwen3_5MLP(nn.Module):
    """Dense SwiGLU MLP（Qwen3.5 非 MoE 层用）。"""

    def __init__(self, config, hidden_size=None, intermediate_size=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        assert config.hidden_act == "silu"
        self.gate_up_proj = MergedColumnParallelLinear(self.hidden_size, [self.intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3_5DecoderLayer(nn.Module):
    """Decoder 层容器（pre-norm 结构，控制流在 Qwen3_5Model 中统一实现）。

    只负责构造本层所需的子模块：
      - linear_attention → Qwen3_5GatedDeltaNet（固定循环状态）
      - full_attention   → Qwen3_5Attention（gated MHA + KV cache）
      - MLP + 两个 Zero-Centered RMSNorm
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            gdn_kwargs = dict(
                hidden_size=config.hidden_size,
                linear_num_value_heads=config.linear_num_value_heads,
                linear_num_key_heads=config.linear_num_key_heads,
                linear_key_head_dim=config.linear_key_head_dim,
                linear_value_head_dim=config.linear_value_head_dim,
                linear_conv_kernel_dim=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_idx=layer_idx,
            )
            self.linear_attn = Qwen3_5GatedDeltaNet(**gdn_kwargs)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Invalid layer_type {self.block_type}")
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3_5Model(nn.Module):
    """Qwen3.5 文本主干。

    管理两类缓存：
    - full attention 层：KV cache（由 Attention 层从 context 的 slot_mapping 读写，与 Qwen3 相同）
    - GDN 层：循环状态池（本模型持有），形状约定
        conv 池: (num_seqs, num_linear, conv_dim, conv_kernel-1)
        rec  池: (num_seqs, num_linear, num_v_heads, head_k_dim, head_v_dim)
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layer_types = config.layer_types
        self.num_linear_layers = _num_linear_layers(config)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i)
                                     for i in range(config.num_hidden_layers)])
        self.norm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # GDN 状态池（由 allocate_mamba_cache 在引擎初始化时创建）
        self.conv_pool: torch.Tensor = torch.empty(0)
        self.rec_pool: torch.Tensor = torch.empty(0)

    def allocate_mamba_cache(self, num_seqs: int) -> None:
        """按最大并发序列数分配 GDN 状态池（对应 vLLM 的 mamba cache 分配）。"""
        c = self.config
        conv_dim = c.linear_key_head_dim * c.linear_num_key_heads * 2 + \
                   c.linear_value_head_dim * c.linear_num_value_heads
        conv_kernel = c.linear_conv_kernel_dim
        # dtype / device 跟随模型参数，而不是依赖调用时的全局默认值
        param = next(self.parameters())
        self.conv_pool = torch.zeros(num_seqs, self.num_linear_layers, conv_dim, conv_kernel - 1,
                                     dtype=param.dtype, device=param.device)
        # 循环状态池用 float32：与 vLLM 的 `mamba_ssm_dtype: float32` 一致
        # （FLA 的 chunk/recurrent 内核本身就按 float32 计算并返回 float32 终态），
        # 池子保持 fp32 可以免去每层每步一次 bf16 回转（24 次 cast + 分配），
        # 同时避免状态被逐 step 舍入到 bf16 造成的精度损失。
        self.rec_pool = torch.zeros(num_seqs, self.num_linear_layers,
                                    c.linear_num_value_heads, c.linear_key_head_dim,
                                    c.linear_value_head_dim,
                                    dtype=torch.float32, device=param.device)

    def _apply_linear_mixer_decode(self, layer, x_normed, li, seq_slots):
        """decode：每个序列恰好 1 个 token，整个 batch 一次算完。

        状态按 seq_slots 在 GPU 上 gather / scatter，全程无 host 同步、无 Python 循环，
        对应 vLLM 的批量 GDN decode 内核。
        """
        slots = seq_slots.long()                                   # (B,)
        conv_state = self.conv_pool[slots, li]                     # (B, C, K-1)
        rec_state = self.rec_pool[slots, li]                       # (B, V, Kd, Vd)
        o, new_conv, new_rec = layer.linear_attn(
            x_normed.unsqueeze(1), conv_state, rec_state)          # (B, 1, D)
        # GDN 内核内部按 float32 计算，返回的循环状态是 float32；状态池是模型 dtype
        # （bf16）。高级索引 scatter 要求 dtype 严格一致，这里显式转型。
        if new_conv.dtype != self.conv_pool.dtype:
            new_conv = new_conv.to(self.conv_pool.dtype)
        if new_rec.dtype != self.rec_pool.dtype:
            new_rec = new_rec.to(self.rec_pool.dtype)
        self.conv_pool[slots, li] = new_conv
        self.rec_pool[slots, li] = new_rec
        return o.squeeze(1)

    def _apply_linear_mixer_prefill(self, layer, x_normed, li, cu_seqlens, slots):
        """prefill：多序列首尾相接，short conv 与 chunk 内核各一次算完。

        状态按 slots 在 GPU 上 gather / scatter，全程无 host 同步、无 Python 循环。
        """
        slots = slots.long()
        rec_state = self.rec_pool[slots, li]                        # (S, V, Kd, Vd)
        o, new_conv, new_rec = layer.linear_attn(
            x_normed.unsqueeze(0), None, rec_state, cu_seqlens=cu_seqlens.long())
        if new_conv.dtype != self.conv_pool.dtype:
            new_conv = new_conv.to(self.conv_pool.dtype)
        if new_rec.dtype != self.rec_pool.dtype:
            new_rec = new_rec.to(self.rec_pool.dtype)
        self.conv_pool[slots, li] = new_conv
        self.rec_pool[slots, li] = new_rec
        return o.squeeze(0)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        cu_seqlens = ctx.cu_seqlens_q
        seq_slots = ctx.seq_slots
        assert cu_seqlens is not None and seq_slots is not None, \
            "需要 ModelRunner 预先 set_context(cu_seqlens=..., seq_slots=...)"

        # 全程在 GPU 上按槽位寻址（prefill 用每序列一个槽位，decode 用每 token 一个）
        is_prefill = ctx.is_prefill
        slots = seq_slots.index_select(0, cu_seqlens[:-1].long()) if is_prefill else None

        hidden_states = self.embed_tokens(input_ids)
        linear_idx = 0
        for layer in self.layers:
            # ---- pre-norm + token mixer（pre-norm 结构，控制流统一在此）----
            x = layer.input_layernorm(hidden_states)
            if layer.block_type == "linear_attention":
                if is_prefill:
                    x = self._apply_linear_mixer_prefill(layer, x, linear_idx, cu_seqlens, slots)
                else:
                    x = self._apply_linear_mixer_decode(layer, x, linear_idx, seq_slots)
                linear_idx += 1
            else:
                x = layer.self_attn(positions, x)
            hidden_states = hidden_states + x          # mixer 残差
            # ---- post-norm + MLP ----
            x = layer.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + layer.mlp(x)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 因果语言模型（加载入口，镜像 vLLM Qwen3_5ForCausalLM）。"""

    # 权重融合映射：加载时把 HF 独立权重焊进融合投影
    # （q/k/v → qkv_proj；gate/up → gate_up_proj）
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        # GDN 输入投影融合（对应 vLLM 的 create_qkvz_proj / create_ba_proj）：
        # qkv+z → in_proj_qkvz，b+a → in_proj_ba
        "in_proj_qkv": ("in_proj_qkvz", 0),
        "in_proj_z": ("in_proj_qkvz", 1),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def allocate_mamba_cache(self, num_seqs: int) -> None:
        """为 GDN 层预分配循环状态池（引擎调度器初始化时调用）。"""
        self.model.allocate_mamba_cache(num_seqs)

    @property
    def conv_pool(self) -> torch.Tensor:
        return self.model.conv_pool

    @property
    def rec_pool(self) -> torch.Tensor:
        return self.model.rec_pool

    def reset_state(self, slot_id: int) -> None:
        """把某个槽位的 GDN 循环状态清零（新序列 / 抢占后重算时调用）。"""
        self.model.conv_pool[slot_id].zero_()
        self.model.rec_pool[slot_id].zero_()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
