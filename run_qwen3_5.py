"""Qwen3.5-4B 端到端推理示例（文本）。

用法：
    python run_qwen3_5.py
    MODEL_PATH=/path/to/model python run_qwen3_5.py
"""
import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/Qwen3.5-4B")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # GDN 状态池按 max_num_seqs 预分配（每序列约 50MB fp32），4B 上不要用默认的 512
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
    )

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=128), use_tqdm=False)
    for prompt, output in zip(prompts, outputs):
        print("=" * 60)
        print("Prompt:", repr(prompt))
        print("Completion:", repr(output["text"]))


if __name__ == "__main__":
    main()
