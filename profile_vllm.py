"""vLLM eager 的 nsys 采样脚本（与 profile_qwen3_5.py 完全同 workload）。

用法：
    nsys profile --force-overwrite=true -o /tmp/nsys_vllm python3 profile_vllm.py
"""
import os
import time

import torch

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams  # noqa: E402

PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/Qwen3.5-4B")
NUM_SEQS = int(os.environ.get("NUM_SEQS", "16"))
IN_LEN = int(os.environ.get("IN_LEN", "512"))
OUT_LEN = int(os.environ.get("OUT_LEN", "128"))


def main():
    llm = LLM(model=PATH, tensor_parallel_size=1, max_num_seqs=NUM_SEQS,
              max_num_batched_tokens=2048, max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=True)

    tok = [[i % 97 + 1] * IN_LEN for i in range(NUM_SEQS)]
    prompts = [{"prompt_token_ids": t} for t in tok]
    warm = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8)] * NUM_SEQS
    full = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=OUT_LEN)] * NUM_SEQS

    llm.generate(prompts, warm)
    torch.cuda.cudart().cudaProfilerStart()
    t = time.time()
    llm.generate(prompts, full)
    elapsed = time.time() - t
    torch.cuda.cudart().cudaProfilerStop()
    total = NUM_SEQS * OUT_LEN
    print(f"vllm eager: {total} tok in {elapsed:.2f}s -> {total / elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
