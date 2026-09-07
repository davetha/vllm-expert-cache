"""ctypes binding to the device-side LRU kernels (liblruexpert.so).

Two entry points, both launched once per MoE layer per forward, ahead of the fused
MoE GEMM. Neither takes a host round-trip, so the whole path stays capturable into a
HIP/CUDA graph -- which is the difference between this cache being fast and being
pinned at eager-mode dispatch cost.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_LIB = None

_SEARCH = (
    lambda: os.environ.get("LRU_CACHE_LIB"),
    lambda: str(Path(__file__).parent / "liblruexpert.so"),
    lambda: "/usr/local/lib/liblruexpert.so",
)


def library_path() -> str:
    for probe in _SEARCH:
        p = probe()
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError(
        "liblruexpert.so not found. Build it with kernels/build.sh for your GPU arch "
        "(e.g. `kernels/build.sh gfx90a`) and either install it next to this package "
        "or point LRU_CACHE_LIB at it."
    )


def lib():
    """Load and bind the kernel library once."""
    global _LIB
    if _LIB is None:
        h = ctypes.CDLL(library_path())
        # manage(topk_ids, n_topk, n_experts, n_slots, max_distinct, max_inserts,
        #        table, map_cold, slot_expert, slot_stamp, routed, step, miss, n_miss,
#        policy, decay, stream)
        h.lru_manage.restype = ctypes.c_int
        h.lru_manage.argtypes = ([ctypes.c_void_p] + [ctypes.c_int] * 5
                                 + [ctypes.c_void_p] * 8 + [ctypes.c_int] * 2
                                 + [ctypes.c_void_p])
        # gather(6x (dst, src, bytes_per_expert), miss, n_miss, chunks, lanes, stream)
        h.lru_gather.restype = ctypes.c_int
        h.lru_gather.argtypes = (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long] * 6
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        )
        _LIB = h
    return _LIB


# The gather kernel copies a fixed six per-expert buffers per call. Quant schemes with
# fewer buffers pad by repeating one they already pass; every slab must be a multiple of
# 16 bytes and dst/src must agree on bytes-per-expert.
GATHER_SLOTS = 6
