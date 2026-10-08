"""量化 nano-vllm 与 vLLM 的 CPU/GPU 时间分解（torch.profiler，无需 nsys）。

用法：
    END=nanovllm python3 analyze_gap.py
    END=vllm     python3 analyze_gap.py

结果写到 /tmp/gap_<END>.json（可用 OUT_JSON 覆盖）。workload 与
profile_qwen3_5.py / profile_vllm.py 保持一致（decode 主导）。
"""
import json
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

END = os.environ.get("END", "nanovllm")
PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/Qwen3.5-4B")
NUM_SEQS = int(os.environ.get("NUM_SEQS", "16"))
IN_LEN = int(os.environ.get("IN_LEN", "512"))
OUT_LEN = int(os.environ.get("OUT_LEN", "128"))
OUT_JSON = os.environ.get("OUT_JSON", "/tmp/gap_%s.json" % END)


def make_engine():
    if END == "vllm":
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams

        llm = LLM(model=PATH, tensor_parallel_size=1, max_num_seqs=NUM_SEQS,
                  max_num_batched_tokens=2048, max_model_len=4096,
                  gpu_memory_utilization=0.85, enforce_eager=True)
        prompts = [{"prompt_token_ids": [i % 97 + 1] * IN_LEN} for i in range(NUM_SEQS)]

        def gen(n):
            sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n)] * NUM_SEQS
            return llm.generate(prompts, sps)

        return llm, gen

    from nanovllm import LLM, SamplingParams

    llm = LLM(PATH, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=NUM_SEQS,
              max_num_batched_tokens=2048, max_model_len=4096, gpu_memory_utilization=0.85)
    prompts = [[i % 97 + 1] * IN_LEN for i in range(NUM_SEQS)]

    def gen(n):
        sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n)] * NUM_SEQS
        return llm.generate(prompts, sps, use_tqdm=False)

    return llm, gen


def main():
    llm, gen = make_engine()
    gen(8)                       # 预热：compile / autotune / 分配
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.time()
        gen(OUT_LEN)
        wall = time.time() - t0
    torch.cuda.synchronize()

    total_gpu = 0.0
    total_cpu = 0.0
    total_count = 0
    rows = []
    for e in prof.key_averages():
        gpu = float(getattr(e, "self_device_time_total", 0) or 0)
        cpu = float(getattr(e, "self_cpu_time_total", 0) or 0)
        rows.append({"key": e.key, "count": int(e.count), "gpu_us": gpu, "cpu_us": cpu})
        total_gpu += gpu
        total_cpu += cpu
        total_count += int(e.count)

    api = {}
    for r in rows:
        if r["key"].startswith("cuda"):
            api[r["key"]] = {"count": r["count"], "cpu_us": round(r["cpu_us"], 1)}

    result = {
        "end": END,
        "wall_s": round(wall, 3),
        "out_tokens": NUM_SEQS * OUT_LEN,
        "tok_per_s": round(NUM_SEQS * OUT_LEN / wall, 1),
        "gpu_kernel_ms": round(total_gpu / 1e3, 1),
        "gpu_busy_ratio": round(total_gpu / 1e6 / wall, 3),
        "self_cpu_ms": round(total_cpu / 1e3, 1),
        "op_count": total_count,
        "cuda_api": api,
        "top_cpu": sorted(rows, key=lambda r: -r["cpu_us"])[:25],
        "top_gpu": sorted(rows, key=lambda r: -r["gpu_us"])[:15],
        "top_count": sorted(rows, key=lambda r: -r["count"])[:15],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    head = {k: v for k, v in result.items()
            if k not in ("top_cpu", "top_gpu", "top_count", "cuda_api")}
    print(json.dumps(head, indent=2))
    print("--- cuda runtime api ---")
    for k, v in sorted(api.items(), key=lambda kv: -kv[1]["count"]):
        print("  %-40s cnt=%-8d cpu_ms=%8.2f" % (k, v["count"], v["cpu_us"] / 1e3))
    print("--- top cpu (self time) ---")
    for r in result["top_cpu"][:20]:
        print("  %-58s cnt=%-8d cpu_ms=%7.1f gpu_ms=%7.1f"
              % (r["key"][:58], r["count"], r["cpu_us"] / 1e3, r["gpu_us"] / 1e3))
    print("--- top gpu kernels ---")
    for r in result["top_gpu"][:12]:
        print("  %-58s cnt=%-8d gpu_ms=%7.1f" % (r["key"][:58], r["count"], r["gpu_us"] / 1e3))

    if END == "nanovllm":
        llm.exit()


if __name__ == "__main__":
    main()
