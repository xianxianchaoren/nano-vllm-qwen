"""Qwen3.5-4B 性能基准（nano-vllm / vLLM 同配置对比）。

用法：
    python bench_qwen3_5.py                 # 默认跑 nano-vllm
    ENGINE=vllm python bench_qwen3_5.py     # 同配置跑 vLLM

可调环境变量（两个引擎共用同一份 workload，保证公平对比）：
    MODEL_PATH, NUM_SEQS, MAX_INPUT_LEN, MAX_OUTPUT_LEN, MAX_MODEL_LEN,
    MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, GPU_MEM_UTIL, EAGER, SEED, VOCAB

注意：
    - 随机 token id 必须小于真实 vocab_size（Qwen3.5 为 248320）。
    - 两边的采样参数完全一致：temperature=0.6、ignore_eos=True、
      max_tokens 逐条相同，因此总产出 token 数严格相等。
"""
import os
import time
from random import randint, seed

ENGINE = os.environ.get("ENGINE", "nanovllm")
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/Qwen3.5-4B")
NUM_SEQS = int(os.environ.get("NUM_SEQS", "64"))
MAX_INPUT_LEN = int(os.environ.get("MAX_INPUT_LEN", "1024"))
MAX_OUTPUT_LEN = int(os.environ.get("MAX_OUTPUT_LEN", "256"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "16"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "2048"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
EAGER = os.environ.get("EAGER", "0") == "1"
SEED = int(os.environ.get("SEED", "0"))
VOCAB = int(os.environ.get("VOCAB", "200000"))
TEMPERATURE = 0.6


def build_workload():
    """固定随机种子，生成与引擎无关的输入，保证两次运行 workload 完全一致。"""
    assert MAX_INPUT_LEN + MAX_OUTPUT_LEN <= MAX_MODEL_LEN, "输入+输出不能超过 max_model_len"
    seed(SEED)
    prompt_token_ids = [
        [randint(0, VOCAB - 1) for _ in range(randint(64, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    output_lens = [randint(1, MAX_OUTPUT_LEN) for _ in range(NUM_SEQS)]
    return prompt_token_ids, output_lens


def run_nanovllm(prompt_token_ids, output_lens):
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        MODEL_PATH,
        enforce_eager=EAGER,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
    )
    sampling_params = [
        SamplingParams(temperature=TEMPERATURE, ignore_eos=True, max_tokens=n)
        for n in output_lens
    ]
    # 预热（触发 torch.compile / 采样器编译），不计入计时
    llm.generate(["Benchmark: "], SamplingParams(temperature=TEMPERATURE, max_tokens=8), use_tqdm=False)

    info = {
        "enforce_eager_effective": bool(llm.model_runner.enforce_eager),
        "num_kvcache_blocks": llm.model_runner.config.num_kvcache_blocks,
    }
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.time() - t
    llm.exit()
    return elapsed, info


def run_vllm(prompt_token_ids, output_lens):
    # 未安装 flashinfer 时，关闭 FlashInfer 采样器（注意力后端用的是 FLASH_ATTN）
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=EAGER,
    )
    sampling_params = [
        SamplingParams(temperature=TEMPERATURE, ignore_eos=True, max_tokens=n)
        for n in output_lens
    ]
    # vLLM 用 dict 形式传 token ids
    prompts = [{"prompt_token_ids": p} for p in prompt_token_ids]
    llm.generate([{"prompt_token_ids": [1]}], SamplingParams(temperature=TEMPERATURE, max_tokens=8))

    t = time.time()
    llm.generate(prompts, sampling_params)
    elapsed = time.time() - t
    return elapsed, {}


def main():
    prompt_token_ids, output_lens = build_workload()
    total_output_tokens = sum(output_lens)
    total_prompt_tokens = sum(len(p) for p in prompt_token_ids)
    runner = run_vllm if ENGINE == "vllm" else run_nanovllm

    elapsed, info = runner(prompt_token_ids, output_lens)

    print("=" * 60)
    print("engine            :", ENGINE)
    print("model             :", MODEL_PATH)
    print("num_seqs          :", NUM_SEQS)
    print("input len         : 64 ..", MAX_INPUT_LEN)
    print("output len        : 1 ..", MAX_OUTPUT_LEN)
    print("max_num_seqs      :", MAX_NUM_SEQS)
    print("max_num_batched_tk:", MAX_NUM_BATCHED_TOKENS)
    print("max_model_len     :", MAX_MODEL_LEN)
    print("enforce_eager     :", EAGER)
    print("eager (effective) :", info.get("enforce_eager_effective", "n/a"))
    if "num_kvcache_blocks" in info:
        print("num_kvcache_blocks:", info["num_kvcache_blocks"])
    print("prompt tokens     :", total_prompt_tokens)
    print("output tokens     :", total_output_tokens)
    print("time (s)          :", round(elapsed, 2))
    print("throughput (tok/s):", round(total_output_tokens / elapsed, 2))
    print("=" * 60)


if __name__ == "__main__":
    main()
