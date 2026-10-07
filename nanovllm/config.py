import os
from dataclasses import dataclass
from transformers import AutoConfig


def resolve_text_config(hf_config):
    """解包多模态 checkpoint 的文本子配置。

    Qwen3.5-4B 的多模态权重是 Qwen3_5ForConditionalGeneration（顶层 model_type
    为 qwen3_5），文本参数挂在 text_config 下（Qwen3_5TextConfig）。本引擎只做
    文本推理，这里统一解包；非多模态 config 原样返回。
    """
    text_config = getattr(hf_config, "text_config", None)
    return hf_config if text_config is None else text_config


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = resolve_text_config(AutoConfig.from_pretrained(self.model))
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
