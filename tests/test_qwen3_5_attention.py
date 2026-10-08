"""Qwen3.5 full-attention 层与 config 解包单测（纯 CPU）。

本机没有 flash_attn，测试注入一个只做形状校验、返回 q 的 Attention stub，
从而在 CPU 上验证 P0 修复：
    1. RoPE 参数从嵌套的 rope_parameters 读取（Qwen3.5-4B 的实际结构）
    2. Qwen3_5Attention.forward 的接线（3D reshape + zero-centered qk-norm +
       partial RoPE + output gate + o_proj）与手算参考值一致
    3. 混合层（GDN + full）模型可前向，reset_state 能清零状态槽
    4. Config 对多模态 text_config 的解包

运行：python -m tests.test_qwen3_5_attention
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

# 绕过 nanovllm/__init__.py（依赖 flash_attn）
_ROOT = Path(__file__).resolve().parent.parent
for _name, _sub in (("nanovllm", ""), ("nanovllm.layers", "layers"),
                    ("nanovllm.models", "models"), ("nanovllm.utils", "utils")):
    _mod = types.ModuleType(_name)
    _mod.__package__ = _name
    _mod.__path__ = [str(_ROOT / "nanovllm" / _sub)]
    sys.modules.setdefault(_name, _mod)

import torch
from torch import nn
import torch.distributed as dist

# 线性层依赖已初始化的进程组（单进程 gloo 足够；显式给 init_method，避免依赖 MASTER_ADDR）
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29517", rank=0, world_size=1)


def _install_attention_stub():
    """注入 CPU 版 Attention：校验形状后返回 q（替代 flash_attn）。"""
    mod = types.ModuleType("nanovllm.layers.attention")

    class Attention(nn.Module):
        def __init__(self, num_heads, head_dim, scale, num_kv_heads):
            super().__init__()
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.scale = scale
            self.k_cache = self.v_cache = torch.tensor([])

        def forward(self, q, k, v):
            assert q.ndim == 3 and q.shape[-1] == self.head_dim, q.shape
            assert k.shape[-1] == self.head_dim and v.shape[-1] == self.head_dim
            assert q.shape[1] == self.num_heads, (q.shape, self.num_heads)
            assert k.shape[1] == self.num_kv_heads and v.shape[1] == self.num_kv_heads
            return q

    mod.Attention = Attention
    sys.modules["nanovllm.layers.attention"] = mod


_install_attention_stub()

from nanovllm.models.qwen3_5 import Qwen3_5Attention, Qwen3_5ForCausalLM
from nanovllm.config import resolve_text_config
from nanovllm.utils.context import set_context, reset_context

torch.manual_seed(0)

# GDN 层的 delta rule 内核由 FLA（Triton）提供，混合层 forward 需要 CUDA
DEVICE = "cuda" if torch.cuda.is_available() else None


def _require_cuda(name):
    if DEVICE is None:
        print(f"SKIP  {name} (需要 CUDA：FLA 内核是 Triton 实现)")
        return False
    return True


def assert_close(a, b, tol=1e-5, msg=""):
    diff = (a.float() - b.float()).abs().max().item()
    assert diff < tol, f"{msg}: max diff = {diff:.3e}"


def build_attn_config():
    # 刻意只给 rope_parameters，不给顶层 rope_theta/partial_rotary_factor，
    # 以此验证实现确实从 rope_parameters 读取（Qwen3.5-4B 的真实结构）
    return SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        attention_bias=False,
        max_position_embeddings=64,
        rope_parameters={"partial_rotary_factor": 0.5, "rope_theta": 10000.0},
    )


def build_model_config():
    layer_types = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
    return SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=53,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        attention_bias=False,
        rope_parameters={"partial_rotary_factor": 0.5, "rope_theta": 10000.0},
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=2,
        tie_word_embeddings=False,
    )


def test_rope_params_from_rope_parameters():
    cfg = build_attn_config()
    attn = Qwen3_5Attention(cfg)
    assert attn.rotary_emb.rotary_dim == int(cfg.head_dim * 0.5), attn.rotary_emb.rotary_dim
    print("PASS  test_rope_params_from_rope_parameters")


def test_gated_attention_forward_matches_reference():
    cfg = build_attn_config()
    attn = Qwen3_5Attention(cfg).to(DEVICE)
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_(torch.randn_like(p) * 0.1)

    n = 5
    x = torch.randn(n, cfg.hidden_size, device=DEVICE)
    positions = torch.arange(n, device=DEVICE)
    out = attn(positions, x)

    # 手算参考：与 forward 语义一致
    qkv = nn.functional.linear(x, attn.qkv_proj.weight)
    q_gate, k, v = qkv.split([attn.q_size * 2, attn.kv_size, attn.kv_size], dim=-1)
    q_gate = q_gate.view(n, attn.num_q_heads, attn.head_dim * 2)
    q, gate = q_gate.chunk(2, dim=-1)
    k = k.view(n, attn.num_kv_heads, attn.head_dim)

    def zero_centered_rms(z, w, eps):
        zf = z.float()
        return (zf * torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).type_as(z)

    q = zero_centered_rms(q, attn.q_norm.weight, cfg.rms_norm_eps)
    k = zero_centered_rms(k, attn.k_norm.weight, cfg.rms_norm_eps)
    q_ref, _ = attn.rotary_emb(positions, q, k)
    o = q_ref * torch.sigmoid(gate)          # stub Attention 返回 q
    ref = nn.functional.linear(o.flatten(1, -1), attn.o_proj.weight)

    assert out.shape == ref.shape
    assert_close(out, ref, 1e-5, "gated attention forward")
    print("PASS  test_gated_attention_forward_matches_reference")


def test_mixed_model_forward_and_reset_state():
    if not _require_cuda("test_mixed_model_forward_and_reset_state"):
        return
    cfg = build_model_config()
    model = Qwen3_5ForCausalLM(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)
    model = model.to(DEVICE)
    model.allocate_mamba_cache(2)

    # 脏数据 -> reset_state 清零
    with torch.no_grad():
        model.conv_pool[1] = 1.0
        model.rec_pool[1] = 1.0
    assert model.conv_pool.shape[0] == 2 and model.rec_pool.shape[0] == 2
    model.reset_state(1)
    assert model.conv_pool[1].abs().sum().item() == 0.0
    assert model.rec_pool[1].abs().sum().item() == 0.0

    tokens = [i % 50 + 1 for i in range(6)]
    ids = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
    pos = torch.arange(6, device=DEVICE)
    cu = torch.tensor([0, 6], dtype=torch.int32, device=DEVICE)
    slots = torch.tensor([0] * 6, dtype=torch.int32, device=DEVICE)
    set_context(True, cu_seqlens_q=cu, seq_slots=slots)
    try:
        hidden = model(ids, pos)
    finally:
        reset_context()

    assert hidden.shape == (6, cfg.hidden_size)
    assert torch.isfinite(hidden).all()
    print("PASS  test_mixed_model_forward_and_reset_state")


def test_decode_shaped_attention_output():
    """回归：decode 时注意力返回 (B, 1, nq, hd)，gate 乘法前必须先 squeeze。"""
    cfg = build_attn_config()
    attn = Qwen3_5Attention(cfg).to(DEVICE)
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_(torch.randn_like(p) * 0.1)

    n = 3
    x = torch.randn(n, cfg.hidden_size, device=DEVICE)
    positions = torch.arange(n, device=DEVICE)

    class DecodeStub(nn.Module):
        # 模拟 flash_attn_with_kvcache 的返回形状
        def forward(self, q, k, v):
            return q.unsqueeze(1)

    attn.attn = DecodeStub()
    out = attn(positions, x)
    assert out.shape == (n, cfg.hidden_size), out.shape

    qkv = nn.functional.linear(x, attn.qkv_proj.weight)
    q_gate, k, v = qkv.split([attn.q_size * 2, attn.kv_size, attn.kv_size], dim=-1)
    q_gate = q_gate.view(n, attn.num_q_heads, attn.head_dim * 2)
    q, gate = q_gate.chunk(2, dim=-1)
    k = k.view(n, attn.num_kv_heads, attn.head_dim)

    def zero_centered_rms(z, w, eps):
        zf = z.float()
        return (zf * torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).type_as(z)

    q = zero_centered_rms(q, attn.q_norm.weight, cfg.rms_norm_eps)
    k = zero_centered_rms(k, attn.k_norm.weight, cfg.rms_norm_eps)
    q_ref, _ = attn.rotary_emb(positions, q, k)
    ref = nn.functional.linear((q_ref * torch.sigmoid(gate)).flatten(1, -1), attn.o_proj.weight)
    assert_close(out, ref, 1e-5, "decode-shaped attention")
    print("PASS  test_decode_shaped_attention_output")


def test_resolve_text_config():
    text = SimpleNamespace(model_type="qwen3_5_text", hidden_size=2560)
    multi = SimpleNamespace(model_type="qwen3_5", text_config=text)
    assert resolve_text_config(multi) is text
    plain = SimpleNamespace(model_type="qwen3", hidden_size=1024)
    assert resolve_text_config(plain) is plain
    print("PASS  test_resolve_text_config")


if __name__ == "__main__":
    test_rope_params_from_rope_parameters()
    test_gated_attention_forward_matches_reference()
    test_mixed_model_forward_and_reset_state()
    test_decode_shaped_attention_output()
    test_resolve_text_config()
    print("\nAll Qwen3.5 attention tests passed.")
