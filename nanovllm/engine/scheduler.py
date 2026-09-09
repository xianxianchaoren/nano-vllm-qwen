from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """调度器：决定每一步（step）把哪些序列放进 GPU batch、每个序列排多少个 token。

    核心策略与 vLLM 一致的两阶段调度：
    - prefill（预填充）：优先处理 waiting 队列，为每个新序列计算 prompt，
      受 max_num_seqs（最大并行序列数）与 max_num_batched_tokens（单步最大 token 数）双重约束。
    - decode（解码）：从 running 队列为每个已就绪序列排 1 个 token（自回归生成）。
    - 抢占（preempt）：KV Cache 块不足时，把部分序列踢回 waiting 队列、释放其块，保证公平。
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs                 # 一个 batch 内最多同时处理的序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 单步最多可排入的 token 总数
        self.eos = config.eos                                   # eos token id，用于判定序列是否结束
        self.block_size = config.kvcache_block_size             # KV Cache 每块容纳的 token 数
        # 块管理器：负责 KV Cache 块的分配/释放/前缀缓存命中
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()   # 等待队列：尚未完成 prefill 的序列
        self.running: deque[Sequence] = deque()   # 运行队列：正在生成（decode）的序列

    def is_finished(self):
        # 两个队列都为空，说明所有请求都已生成完毕
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 新请求进入等待队列，等待被调度
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """产出一批可执行的序列。返回值：(本批序列列表, 是否为 prefill 阶段)。

        - 只要 waiting 非空就优先做 prefill（填满本步的 token 预算）；
        - prefill 没排到东西时，才对 running 队列做 decode。
        """
        scheduled_seqs = []          # 本批被选中的序列
        num_batched_tokens = 0       # 本步已累计的 token 数（不能超过 max_num_batched_tokens）

        # ========== prefill 阶段 ==========
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens  # 本步剩余可用的 token 预算
            if remaining == 0:
                break  # 预算耗尽，停止排入

            if not seq.block_table:
                # 新序列（还没有分配 KV 块）：先检查前缀缓存命中了多少个块
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break  # 空闲块不足，无法为新序列分配，等后续步骤（可能先做 decode）
                # 本次实际需要计算（排入）的 token 数 = 总 token 数 - 缓存直接复用的 token 数
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 已被抢占/分块过的序列：只需排"尚未缓存"的剩余部分
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 若剩余预算装不下当前序列的全部 token，且 batch 里已有其他序列，
            # 则中断（chunked prefill 只允许 batch 中第一个序列分块，避免碎片化）
            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                # 正式分配 KV 块（含复用命中的缓存块）
                self.block_manager.allocate(seq, num_cached_blocks)
            # 本次排入的 token 数 = min(还需计算的 token, 剩余预算)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # 若缓存过的 token + 本次排入的 token 覆盖了整个 prompt，说明 prefill 完成，转为 RUNNING
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()   # 移出等待队列
                self.running.append(seq) # 进入运行队列（后续参与 decode）
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True  # 本步执行 prefill

        # ========== decode 阶段（没有 prefill 任务时才执行） ==========
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 若当前序列需要新块但空闲块不足，则抢占其他序列释放块；
            # while...else：循环正常结束（无 break）时进入 else 分支，表示块分配成功
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())  # 抢占队尾序列，回收它的块
                else:
                    self.preempt(seq)                 # 没有别的序列可抢，只能抢占它自己
                    break
            else:
                seq.num_scheduled_tokens = 1   # decode 阶段每序列每步只排 1 个 token
                seq.is_prefill = False
                self.block_manager.may_append(seq)  # 若跨块边界则追加一个新块
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 恢复 running 队列顺序（popleft 的顺序，让已排入的序列排到队首）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # 本步执行 decode

    def preempt(self, seq: Sequence):
        """抢占：释放序列占用的全部 KV 块，将其退回 waiting 队列（之后需重新 prefill）。"""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True                 # 重新进入时按 prefill 处理
        self.block_manager.deallocate(seq)    # 释放全部块（块内容丢弃，重新计算）
        self.waiting.appendleft(seq)          # 插回队首，保证被抢占的序列优先被调度

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """模型前向完成后回调：推进序列状态、更新前缀缓存、判断是否结束。"""
        for seq, token_id in zip(seqs, token_ids):
            # 更新本批新写入的 KV 块的哈希（供前缀缓存匹配），并登记到 hash 表
            self.block_manager.hash_blocks(seq)
            # 推进"已缓存 token 计数"，重置本次排入的 token 数
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue  # chunked prefill：prompt 还没算完，本步不产出新 token，直接进入下一步

            # 追加模型新生成的 token（同时使 num_tokens +1）
            seq.append_token(token_id)
            # 终止条件：1) 生成 eos（且未设置 ignore_eos） 2) 达到 max_tokens 上限
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 释放序列占用的全部 KV 块
                self.running.remove(seq)            # 移出运行队列
