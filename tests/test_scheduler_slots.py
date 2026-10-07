"""调度器槽位生命周期单测（纯 CPU，无需 torch/CUDA）。

覆盖 GDN 状态池槽位（Sequence.slot_id）的分配/回收：
    1. prefill 为每个序列分配唯一槽位，序列结束时归还
    2. 因 token 预算不足而中断时，未调度序列的槽位必须归还（无泄漏）
    3. 槽位耗尽时不崩溃
    4. 抢占时归还槽位与 KV 块

运行：python -m tests.test_scheduler_slots
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------
# 绕过 nanovllm/__init__.py（它 import LLMEngine -> transformers/flash_attn，
# 本机轻量环境没有 flash_attn）。用带 __path__ 的空壳包直接加载子模块。
# ---------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
for _name, _sub in (("nanovllm", ""), ("nanovllm.engine", "engine")):
    _mod = types.ModuleType(_name)
    _mod.__package__ = _name
    _mod.__path__ = [str(_ROOT / "nanovllm" / _sub)]
    sys.modules.setdefault(_name, _mod)

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler


def make_scheduler(max_num_seqs, max_num_batched_tokens, num_kvcache_blocks, layer_types):
    Sequence.block_size = 256
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=num_kvcache_blocks,
        hf_config=SimpleNamespace(layer_types=layer_types),
    )
    return Scheduler(config)


def test_prefill_assigns_and_finish_releases_slots():
    sch = make_scheduler(4, 16384, 64, ["linear_attention"])
    sp = SamplingParams(temperature=0.6, max_tokens=1)
    seqs = [Sequence([1] * 10, sp) for _ in range(3)]
    for seq in seqs:
        sch.add(seq)

    scheduled, is_prefill = sch.schedule()
    assert is_prefill and scheduled == seqs
    slots = [seq.slot_id for seq in scheduled]
    assert all(s is not None for s in slots)
    assert len(set(slots)) == 3, f"槽位必须互不相同: {slots}"
    assert len(sch.free_slots) == 1

    # 生成结束（max_tokens=1）-> 归还槽位 + KV 块
    sch.postprocess(scheduled, [2, 3, 4], is_prefill=True)
    assert all(seq.is_finished for seq in scheduled)
    assert all(seq.slot_id is None for seq in scheduled)
    assert len(sch.free_slots) == 4
    assert len(sch.block_manager.free_block_ids) == 64
    assert sch.is_finished()
    print("PASS  test_prefill_assigns_and_finish_releases_slots")


def test_no_slot_leak_on_budget_break():
    # 预算 8：seq_a 排入 5 个 token 后只剩 3，seq_b 无法整段排入 -> 必须归还槽位
    sch = make_scheduler(4, 8, 64, ["linear_attention"])
    sp = SamplingParams(temperature=0.6, max_tokens=4)
    a, b = Sequence([1] * 5, sp), Sequence([2] * 5, sp)
    sch.add(a)
    sch.add(b)

    scheduled, is_prefill = sch.schedule()
    assert is_prefill and scheduled == [a]
    assert a.slot_id is not None
    assert b.slot_id is None, "未调度序列的槽位必须归还"
    assert b.block_table == []
    assert len(sch.free_slots) == 3
    print("PASS  test_no_slot_leak_on_budget_break")


def test_no_slots_available_no_crash():
    # max_num_seqs=1：a 已占满唯一槽位，此时再排 b 不能崩溃
    sch = make_scheduler(1, 1, 64, ["linear_attention"])
    sp = SamplingParams(temperature=0.6, max_tokens=4)
    a, b = Sequence([1], sp), Sequence([2], sp)
    sch.add(a)
    scheduled, is_prefill = sch.schedule()
    assert is_prefill and scheduled == [a]
    assert len(sch.free_slots) == 0

    sch.add(b)
    scheduled, is_prefill = sch.schedule()   # 槽位耗尽，应回退到 decode 而不是抛异常
    assert scheduled == [a] and is_prefill is False
    assert b.slot_id is None
    assert b.status.name == "WAITING"
    print("PASS  test_no_slots_available_no_crash")


def test_preempt_releases_slot():
    # 块总数恰好被 a(1 块) + b(2 块) 用满；b 跨块时需要抢占
    sch = make_scheduler(2, 16384, 3, ["linear_attention"])
    sp = SamplingParams(temperature=0.6, max_tokens=4)
    a, b = Sequence([1] * 256, sp), Sequence([2] * 257, sp)
    sch.add(a)
    sch.add(b)

    scheduled, is_prefill = sch.schedule()
    assert is_prefill and scheduled == [a, b]
    assert len(sch.block_manager.free_block_ids) == 0
    assert len(sch.free_slots) == 0

    scheduled, is_prefill = sch.schedule()
    assert is_prefill is False and scheduled == [a]
    assert b.status.name == "WAITING" and b.block_table == []
    assert b.slot_id is None, "被抢占序列的槽位必须归还"
    assert len(sch.free_slots) == 1
    print("PASS  test_preempt_releases_slot")


if __name__ == "__main__":
    test_prefill_assigns_and_finish_releases_slots()
    test_no_slot_leak_on_budget_break()
    test_no_slots_available_no_crash()
    test_preempt_releases_slot()
    print("\nAll scheduler slot tests passed.")
