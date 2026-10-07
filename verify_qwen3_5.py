"""Qwen3.5-4B 输出正确性校验：nano-vllm vs HF transformers 参考实现。

做法：对多段 prompt，分别取最后一个位置的 logits（全词表），对比
    - 最大绝对误差
    - top-1 token 是否一致
    - top-5 集合重叠度

用法：python verify_qwen3_5.py
"""
import os

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/Qwen3.5-4B")

PROMPTS = [
    "What is the capital of France? Answer briefly.",
    "Write a Python function that reverses a string.",
    "Explain what a prime number is in one sentence.",
]


def build_inputs(tokenizer):
    ids_list = []
    for prompt in PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, return_tensors="pt").input_ids
        assert ids.shape[1] <= 256, "prompt 必须短于 256 个 token（单块 KV）"
        ids_list.append(ids)
    return ids_list


def hf_logits_all(ids_list, dtype):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=dtype)
    model.to("cuda").eval()
    outs = []
    with torch.inference_mode():
        for ids in ids_list:
            hidden = model.model.language_model(input_ids=ids.cuda(), use_cache=False)
            logits = model.lm_head(hidden.last_hidden_state)
            outs.append(logits[0, -1].float().cpu())
    del model
    torch.cuda.empty_cache()
    return outs


def nano_logits_all(ids_list):
    from nanovllm import LLM
    from nanovllm.utils.context import set_context, reset_context

    llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1,
              max_num_seqs=4, max_num_batched_tokens=512, max_model_len=512,
              gpu_memory_utilization=0.8)
    model = llm.model_runner.model
    outs = []
    for ids in ids_list:
        flat = ids[0].cuda()
        n = int(flat.shape[0])
        cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        slot_mapping = torch.arange(n, dtype=torch.int32, device="cuda")
        seq_slots = torch.zeros(n, dtype=torch.int32, device="cuda")
        positions = torch.arange(n, dtype=torch.int64, device="cuda")

        model.reset_state(0)      # warmup 会写脏状态池槽位
        set_context(True, cu, cu, n, n, slot_mapping, None, None, seq_slots)
        try:
            with torch.inference_mode():
                hidden = model(flat, positions)
                logits = model.compute_logits(hidden)
        finally:
            reset_context()
        outs.append(logits[0].float().cpu())
    llm.exit()
    return outs


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids_list = build_inputs(tokenizer)

    print("--- HF reference ---")
    refs_bf16 = hf_logits_all(ids_list, torch.bfloat16)
    print("--- HF reference (fp32, 高精度参照) ---")
    refs_fp32 = hf_logits_all(ids_list, torch.float32)
    print("--- nano-vllm ---")
    gots = nano_logits_all(ids_list)

    all_ok = True
    print("\n--- compare ---")
    for i in range(len(ids_list)):
        ref16, ref32, got = refs_bf16[i], refs_fp32[i], gots[i]
        top1_bf16, top1_fp32, top1_got = int(ref16.argmax()), int(ref32.argmax()), int(got.argmax())
        top5_fp32 = set(ref32.topk(5).indices.tolist())
        top5_got = set(got.topk(5).indices.tolist())
        overlap = len(top5_fp32 & top5_got)
        # fp32 下 top1 与 top2 的间距：间距很小说明是数值平局
        margin = float(ref32.topk(2).values[0] - ref32.topk(2).values[1])
        ok = (top1_fp32 == top1_got) and overlap == 5
        all_ok = all_ok and ok
        tag = "MATCH" if ok else "MISMATCH"
        decoded = tokenizer.decode([top1_got])
        print(f"prompt {i}: tokens={ids_list[i].shape[1]} "
              f"max_diff_bf16={(ref16 - got).abs().max().item():.4f} "
              f"top1_bf16={top1_bf16} top1_fp32={top1_fp32} top1_nano={top1_got} "
              f"top5_overlap={overlap}/5 margin_fp32={margin:.4f} {tag} token={decoded!r}")

    print("\nRESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
