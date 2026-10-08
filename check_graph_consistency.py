"""一致性检查：CUDA graph 路径与 eager 路径必须产出完全相同的 token。

同一个 workload、同一个随机种子下跑两遍（enforce_eager=True / False），
逐 token 比较。若 graph 路径存在 stale 输入、状态池被 padding 行污染等问题，
这里会立刻暴露。

用法（两次独立进程，避免显存未释放）：
    MODE=eager python check_graph_consistency.py
    MODE=graph python check_graph_consistency.py
    python check_graph_consistency.py          # 比较两次结果
"""
import json
import os

import torch

from nanovllm import LLM, SamplingParams

PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/Qwen3.5-4B")
NUM_SEQS = int(os.environ.get("N", "5"))
MAX_NEW = int(os.environ.get("MAX_NEW", "48"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "8"))


def run(eager):
    torch.manual_seed(0)
    llm = LLM(
        PATH,
        enforce_eager=eager,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        gpu_memory_utilization=0.6,
    )
    prompts = [[i % 97 + 1] * 64 for i in range(NUM_SEQS)]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=MAX_NEW)
        for _ in range(NUM_SEQS)
    ]
    llm.generate(["warmup"], SamplingParams(temperature=0.6, max_tokens=4), use_tqdm=False)
    torch.manual_seed(0)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    tokens = [tuple(o["token_ids"]) for o in outputs]
    llm.exit()
    return tokens


def main():
    mode = os.environ.get("MODE", "compare")
    if mode in ("eager", "graph"):
        tokens = run(mode == "eager")
        with open(f"/tmp/tokens_{mode}.json", "w") as f:
            json.dump([list(t) for t in tokens], f)
        print(f"{mode} done: {len(tokens)} seqs x {len(tokens[0])} tokens")
        return

    with open("/tmp/tokens_eager.json") as f:
        eager_tokens = [tuple(t) for t in json.load(f)]
    with open("/tmp/tokens_graph.json") as f:
        graph_tokens = [tuple(t) for t in json.load(f)]

    same = eager_tokens == graph_tokens
    for i, (a, b) in enumerate(zip(eager_tokens, graph_tokens)):
        if a != b:
            pos = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])
            print(f"seq {i}: 第 {pos} 个 token 起不一致 (eager={a[pos:pos+4]} graph={b[pos:pos+4]})")
    print("token 完全一致:", same)
    print("RESULT:", "PASS" if same else "FAIL")


if __name__ == "__main__":
    main()
