import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


_NON_TEXT_PREFIXES = ("model.visual.", "visual.", "mtp.")


def resolve_weight_name(name: str) -> str | None:
    """把 checkpoint 的权重名映射到本引擎的模块路径。

    Qwen3.5-4B 是 Qwen3_5ForConditionalGeneration：文本参数在
    `model.language_model.*` 下，另有视觉（`model.visual.*`）与 MTP（`mtp.*`）
    权重。本引擎只做文本推理，这里把文本前缀改写为 `model.*`，其余返回 None 跳过。
    """
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    for prefix in _NON_TEXT_PREFIXES:
        if name.startswith(prefix):
            return None
    return name


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                name = resolve_weight_name(weight_name)
                if name is None:
                    continue
                for k in packed_modules_mapping:
                    if k in name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param_name = name
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                loaded.add(param_name)
    # 兜底校验：任何未被加载的参数（tie 的 lm_head 除外）都说明映射有误
    missing = [n for n in model.state_dict() if n not in loaded and not n.endswith("lm_head.weight")]
    assert not missing, f"以下权重未从 checkpoint 加载: {missing[:8]}"
