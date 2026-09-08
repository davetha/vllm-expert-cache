"""Drive the real manage kernel with identical routing under each policy and count misses.

Deterministic and serve-free, so it measures the policy itself rather than throughput noise.
A working LFU should take FEWER inserts than LRU on skewed / interleaved routing.
"""
import ctypes, os, sys
import numpy as np
import torch

LIB = os.environ.get("EXPERT_CACHE_LIB", "/repo/vllm_expert_cache/libexpertcache.so")
lib = ctypes.CDLL(LIB)
lib.expert_cache_manage.restype = ctypes.c_int
lib.expert_cache_manage.argtypes = ([ctypes.c_void_p] + [ctypes.c_int] * 5
                           + [ctypes.c_void_p] * 8 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
DEV = "cuda:0"


class State:
    def __init__(self, E, S):
        self.E, self.S = E, S
        self.table = torch.full((E,), -1, dtype=torch.int32, device=DEV)
        self.map_cold = torch.arange(E, dtype=torch.int32, device=DEV)
        self.slot_expert = torch.full((S,), -1, dtype=torch.int32, device=DEV)
        self.slot_stamp = torch.zeros((S,), dtype=torch.int64, device=DEV)
        self.routed = torch.zeros((E,), dtype=torch.uint8, device=DEV)
        self.step = torch.zeros((1,), dtype=torch.int64, device=DEV)
        self.miss = torch.zeros((2 * S,), dtype=torch.int32, device=DEV)
        self.n_miss = torch.zeros((1,), dtype=torch.int32, device=DEV)


def run(trace, E, S, policy, decay=64):
    s = State(E, S)
    total = 0
    for ids in trace:
        t = torch.tensor(ids, dtype=torch.int32, device=DEV)
        rc = lib.expert_cache_manage(
            ctypes.c_void_p(t.data_ptr()), t.numel(), E, S,
            min(E, t.numel()), S,
            ctypes.c_void_p(s.table.data_ptr()), ctypes.c_void_p(s.map_cold.data_ptr()),
            ctypes.c_void_p(s.slot_expert.data_ptr()), ctypes.c_void_p(s.slot_stamp.data_ptr()),
            ctypes.c_void_p(s.routed.data_ptr()), ctypes.c_void_p(s.step.data_ptr()),
            ctypes.c_void_p(s.miss.data_ptr()), ctypes.c_void_p(s.n_miss.data_ptr()),
            policy, decay,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        assert rc == 0, rc
        torch.cuda.synchronize()
        total += int(s.n_miss.item())
    return total


def zipf_trace(rng, steps, E, k, a=1.2):
    """Single stream: skewed routing, the shape MoE routers actually produce."""
    w = 1.0 / np.power(np.arange(1, E + 1), a)
    w /= w.sum()
    perm = rng.permutation(E)
    return [perm[rng.choice(E, size=k, replace=False, p=w)] for _ in range(steps)]


def interleaved_trace(rng, steps, E, k, ntask=4, a=1.2, shared=0.5):
    """Several tasks round-robin: each task has its own hot set, partly shared."""
    w = 1.0 / np.power(np.arange(1, E + 1), a); w /= w.sum()
    nshare = int(E * shared)
    shared_pool = rng.permutation(E)[:nshare]
    tasks = []
    for _ in range(ntask):
        own = rng.permutation(E)[:E - nshare]
        tasks.append(np.concatenate([shared_pool, own]))
    out = []
    for i in range(steps):
        pool = tasks[i % ntask]
        idx = rng.choice(len(pool), size=k, replace=False, p=w[:len(pool)] / w[:len(pool)].sum())
        out.append(pool[idx])
    return out


if __name__ == "__main__":
    E, k, steps = 128, 8, 600
    print(f"E={E} top_k={k} steps={steps}   (misses = experts fetched over PCIe)\n")
    for name, gen in (("single-stream zipf", zipf_trace),
                      ("4 tasks interleaved", interleaved_trace)):
        print(f"--- {name} ---")
        for S in (16, 32, 64):
            rng = np.random.default_rng(1234)
            trace = gen(rng, steps, E, k)
            lru = run(trace, E, S, 0)
            lfu = run(trace, E, S, 1)
            d = (lfu - lru) / max(lru, 1) * 100
            verdict = "LFU better" if lfu < lru else ("tie" if lfu == lru else "LFU WORSE")
            print(f"  slots={S:3d}   LRU {lru:6d}   LFU {lfu:6d}   {d:+6.1f}%   {verdict}")
        print()
