"""Per-MoE-layer expert slot cache.

The model's full expert tensors stay where the offloader put them -- host RAM, mapped so
the GPU can read them. This class adds a small set of VRAM *slots* and an LRU that decides
which experts occupy them. Each forward:

  1. `lru_manage` marks the experts routed this step, refreshes their stamps, evicts the
     least-recently-routed slots that this step does NOT need, and emits a miss list.
  2. `lru_gather` pulls each missing expert's rows from host memory into its new slot.
  3. The routing ids are remapped expert -> slot, and the stock fused-MoE kernel runs
     against the slot buffers.

Nothing here reads a device value back to the host, so the sequence is graph-capturable.
Both tensor-parallel ranks see identical routing and the kernels are deterministic, so the
caches evolve identically across ranks without any synchronisation.
"""

from __future__ import annotations

import ctypes
import os

import torch

from ._lib import GATHER_SLOTS, lib


def _bytes_per_expert(t: torch.Tensor) -> int:
    return (t.numel() // t.shape[0]) * t.element_size()


class ExpertSlotCache:
    """Holds `num_slots` of `num_experts` experts resident in device memory."""

    def __init__(self, owner, source_names, num_experts: int, num_slots: int, device):
        # Sources are read fresh from `owner` on every step: vLLM's expert offloader
        # relocates these tensors to host memory *after* weight loading, so a pointer
        # captured at setup time would dangle.
        self.owner = owner
        self.names = list(source_names)
        self.E = int(num_experts)
        self.S = int(num_slots)

        self.slots = {}
        for n in self.names:
            full = getattr(owner, n).data
            self.slots[n] = torch.empty(
                (self.S,) + tuple(full.shape[1:]), dtype=full.dtype, device=device
            )

        i32, i64, u8 = torch.int32, torch.int64, torch.uint8
        self.table = torch.full((self.E,), -1, dtype=i32, device=device)   # expert -> slot
        self.map_cold = torch.arange(self.E, dtype=i32, device=device)     # expert -> itself if cold
        self.slot_expert = torch.full((self.S,), -1, dtype=i32, device=device)
        self.slot_stamp = torch.zeros((self.S,), dtype=i64, device=device)
        self.routed = torch.zeros((self.E,), dtype=u8, device=device)
        self.step = torch.zeros((1,), dtype=i64, device=device)
        self.miss = torch.zeros((2 * self.S,), dtype=i32, device=device)
        self.n_miss = torch.zeros((1,), dtype=i32, device=device)

        self.chunks = int(os.environ.get("VLLM_LRU_CHUNKS", "16"))
        self.lanes = int(os.environ.get("VLLM_LRU_LANES", "64"))

    def fits(self, topk_ids: torch.Tensor) -> bool:
        """Can this step be served entirely from slots?

        `topk_ids.numel()` is tokens x top_k -- an upper bound on the distinct experts the
        step can touch, and pure shape metadata, so this costs no device sync. Decode
        (one token) always fits; wide prefill steps do not and read through instead.
        """
        return topk_ids.numel() <= self.S

    def remap(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Rewrite routing ids as slot ids. Valid only after `refresh`."""
        return self.table[topk_ids.to(torch.int64)].to(topk_ids.dtype)

    def refresh(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Run the manager + gather for this step and return slot-space routing ids."""
        h = lib()
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        ids = topk_ids.to(torch.int32).contiguous()

        rc = h.lru_manage(
            ctypes.c_void_p(ids.data_ptr()), int(ids.numel()), self.E, self.S,
            min(self.E, int(ids.numel())), self.S,
            ctypes.c_void_p(self.table.data_ptr()),
            ctypes.c_void_p(self.map_cold.data_ptr()),
            ctypes.c_void_p(self.slot_expert.data_ptr()),
            ctypes.c_void_p(self.slot_stamp.data_ptr()),
            ctypes.c_void_p(self.routed.data_ptr()),
            ctypes.c_void_p(self.step.data_ptr()),
            ctypes.c_void_p(self.miss.data_ptr()),
            ctypes.c_void_p(self.n_miss.data_ptr()),
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"lru_manage failed rc={rc}")

        dsts, srcs = [], []
        for n in self.names:
            dsts.append(self.slots[n])
            srcs.append(getattr(self.owner, n).data)
        # The gather kernel always copies six buffers; pad by repeating the last real one
        # (a redundant but valid copy) rather than passing unreadable memory.
        while len(dsts) < GATHER_SLOTS:
            dsts.append(dsts[-1])
            srcs.append(srcs[-1])

        args = []
        for d, s in zip(dsts, srcs):
            args += [ctypes.c_void_p(d.data_ptr()), ctypes.c_void_p(s.data_ptr()),
                     ctypes.c_long(_bytes_per_expert(s))]
        rc = h.lru_gather(
            *args,
            ctypes.c_void_p(self.miss.data_ptr()),
            ctypes.c_void_p(self.n_miss.data_ptr()),
            self.chunks, self.lanes, stream,
        )
        if rc != 0:
            raise RuntimeError(f"lru_gather failed rc={rc} (slab sizes must be 16B multiples)")

        return self.remap(topk_ids)
