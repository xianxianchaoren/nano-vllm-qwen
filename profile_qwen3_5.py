"""nsys 用采样脚本：固定长度、decode 主导的 workload。

用法：
    nsys profile --stats=true --force-overwrite=true -o /tmp/nsys_prof \
        python3 profile_qwen3_5.py
"""
import os
import time

import torch

from nanovllm import LLM, SamplingParams

PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/Qwen3.5-4B")
NUM_SEQS = int(os.environ.get("NUM_SEQS", "16"))
IN_LEN = int(os.environ.get("IN_LEN", "512"))
OUT_LEN = int(os.environ.get("OUT_LEN", "128"))


def main():
    llm = LLM(PATH, enforce_eager=True, tensor_parallel_size=1,
              max_num_seqs=NUM_SEQS, max_num_batched_tokens=2048,
              max_model_len=4096, gpu_memory_utilization=0.85)

    prompts = [[i % 97 + 1] * IN_LEN for i in range(NUM_SEQS)]
    warm = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8)] * NUM_SEQS
    full = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=OUT_LEN)] * NUM_SEQS

    # 预热：compile / autotune 全部跑掉，并让调度器进入 decode 稳态
    llm.generate(prompts, warm, use_tqdm=False)

    torch.cuda.cudart().cudaProfilerStart()
    t = time.time()
    llm.generate(prompts, full, use_tqdm=False)
    elapsed = time.time() - t
    torch.cuda.cudart().cudaProfilerStop()
    total = NUM_SEQS * OUT_LEN
    print(f"decode-heavy: {total} tok in {elapsed:.2f}s -> {total / elapsed:.1f} tok/s")
    llm.exit()


if __name__ == "__main__":
    main()
