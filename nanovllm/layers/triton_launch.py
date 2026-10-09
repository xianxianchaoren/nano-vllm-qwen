"""Triton 启动器缓存（把「编译产物 + 预绑定启动器」复用起来）。

背景：`jit_fn[grid](**kwargs)` 每次调用都要走 binder —— 解析 kwargs、计算特化
key、查编译缓存，实测 ~16us/次；还要叠加各包装层自己的 shape/stride 计算与临时
张量分配，一次很简单的 norm 调用就会到 40us 以上。

Qwen3.5 的 decode 每步有上百次这类小 kernel 调用（layer norm、gated norm、
short conv、FLA recurrent），在 eager 模式下这些 Python/launch 开销就是瓶颈
（GPU 利用率只有 ~30%）。这里把编译好的 kernel 与预绑定启动器按
「shape / dtype / stride / constexpr / num_warps」缓存下来，命中后只剩一次普通
函数调用（~9us），是 eager 追平 vLLM 的关键手段之一。

注意：`args` 必须与 kernel 形参顺序严格一致（constexpr 参数除外），且 `key`
必须覆盖所有会影响 Triton 特化的信息，否则复用到不匹配的 cubin 会算错。
"""
from __future__ import annotations

from functools import lru_cache

import triton

_LAUNCHERS: dict = {}


@lru_cache(maxsize=None)
def block_n(n: int, element_size: int) -> int:
    """每行一个 program 时覆盖一行所需的 2 次幂宽度（受 shared memory 上限约束）。"""
    return min(65536 // element_size, triton.next_power_of_2(n))


def bound_launch(jit_fn, tag, key, grid, *args, constexprs, num_warps, num_stages=3):
    """返回 (预绑定启动器, 位置参数元组)；未命中缓存时先编译（不执行）。

    Args:
        jit_fn: `triton.jit` 内核，或 `triton.heuristics` 包装后的对象
            （两者都有 `warmup`，后者内部会自动补上 heuristic 求出的 constexpr）
        key: 特化相关的全部信息（shape / dtype / stride / constexpr / num_warps）
    """
    cache_key = (tag,) + tuple(key)
    runner = _LAUNCHERS.get(cache_key)
    if runner is None:
        compiled = jit_fn.warmup(*args, **constexprs, num_warps=num_warps,
                                 num_stages=num_stages, grid=grid)
        runner = compiled[(grid[0], grid[1] if len(grid) > 1 else 1, 1)]
        _LAUNCHERS[cache_key] = runner
    return runner, args
