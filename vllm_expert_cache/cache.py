"""Per-MoE-layer expert slot cache.

The model's full expert tensors stay where the offloader put them -- host RAM, mapped so
the GPU can read them. This class adds a small set of VRAM *slots* and a policy that decides
which experts occupy them. Each forward:

  1. `expert_cache_manage` marks the experts routed this step, refreshes their priorities,
     evicts the lowest-priority slots that this step does NOT need, and emits a miss list.
  2. `expert_cache_gather` pulls each missing expert's rows from host memory into its new slot.
  3. The routing ids are remapped expert -> slot, and the stock fused-MoE kernel runs
     against the slot buffers.

Nothing here reads a device value back to the host, so the sequence is graph-capturable.
Both tensor-parallel ranks see identical routing and the kernels are deterministic, so the
caches evolve identically across ranks without any synchronisation.

The set of mirrored tensors is whatever the caller passes: two for an unquantized model,
four for int8 (weights plus their scales), six for a scheme that also carries zero points.
The gather kernel copies a fixed `GATHER_SLOTS` buffers per launch, so wider sets are split
across several launches rather than needing a wider kernel.
"""

from __future__ import annotations

import ctypes
import os

import torch

from ._lib import GATHER_SLOTS, lib
from .config import settings


def _bytes_per_expert(t: torch.Tensor) -> int:
    return (t.numel() // t.shape[0]) * t.element_size()


class ExpertSlotCache:
    """Holds `num_slots` of `num_experts` experts resident in device memory."""

    def __init__(self, owner, source_names, num_experts: int, num_slots: int, device,
                 required=()):
        # Sources are read fresh from `owner` on every step: vLLM's expert offloader
        # relocates these tensors to host memory *after* weight loading, so a pointer
        # captured at setup time would dangle.
        self.owner = owner
        self.names = list(source_names)
        self.E = int(num_experts)
        self.S = int(num_slots)

        self.slots = {}
        self.skipped = []
        ragged = []
        for n in self.names:
            full = getattr(owner, n).data
            # The gather moves 16 bytes per lane, so a per-expert slab that is not a
            # multiple of 16 would have its tail dropped -- silently, since the kernel
            # only reports launch failures.
            #
            # A ragged *weight* is fatal: the fused kernel is handed it directly. A ragged
            # anything-else is usually bookkeeping (wNa16 registers a 2-element
            # `w13_weight_shape` holding the original dims), so skip mirroring it and let
            # the caller's leak check decide -- if the quantisation config really does
            # index it per expert, that check refuses the layer; if it does not, mirroring
            # it was never needed.
            if _bytes_per_expert(full) % 16:
                (ragged if n in required else self.skipped).append(
                    f"{n}{tuple(full.shape)}={_bytes_per_expert(full)}B")
                continue
            self.slots[n] = torch.empty(
                (self.S,) + tuple(full.shape[1:]), dtype=full.dtype, device=device
            )
        if ragged:
            raise ValueError(
                "per-expert slab must be a multiple of 16 bytes, but these are not: "
                + ", ".join(ragged))
        self.names = [n for n in self.names if n in self.slots]

        # What the layer's attributes are rebound to during a cached step. Built once:
        # vLLM captures parameter storage addresses into CUDA/HIP graphs, and churning
        # Parameter objects on the hot path is wasted work besides.
        self.bound: dict[str, torch.Tensor] = dict(self.slots)

        i32, i64, u8 = torch.int32, torch.int64, torch.uint8
        self.table = torch.full((self.E,), -1, dtype=i32, device=device)   # expert -> slot
        self.map_cold = torch.arange(self.E, dtype=i32, device=device)     # expert -> itself if cold
        self.slot_expert = torch.full((self.S,), -1, dtype=i32, device=device)
        self.slot_stamp = torch.zeros((self.S,), dtype=i64, device=device)
        self.routed = torch.zeros((self.E,), dtype=u8, device=device)
        self.step = torch.zeros((1,), dtype=i64, device=device)
        self.miss = torch.zeros((2 * self.S,), dtype=i32, device=device)
        self.n_miss = torch.zeros((1,), dtype=i32, device=device)

        self.policy = settings.policy_code
        self.decay = settings.decay
        self.chunks = int(os.environ.get("EXPERT_CACHE_CHUNKS", "16"))
        self.lanes = int(os.environ.get("EXPERT_CACHE_LANES", "64"))

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

        rc = h.expert_cache_manage(
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
            self.policy, self.decay,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"expert_cache_manage failed rc={rc}")

        dsts, srcs = [], []
        for n in self.names:
            dsts.append(self.slots[n])
            srcs.append(getattr(self.owner, n).data)

        # The gather kernel copies a fixed GATHER_SLOTS buffers per launch. Split wider
        # sets across launches, and pad the last group by repeating a buffer it already
        # carries (a redundant but valid copy) rather than passing unreadable memory.
        for i in range(0, len(dsts), GATHER_SLOTS):
            gd = dsts[i:i + GATHER_SLOTS]
            gs = srcs[i:i + GATHER_SLOTS]
            while len(gd) < GATHER_SLOTS:
                gd.append(gd[-1])
                gs.append(gs[-1])

            args = []
            for d, s in zip(gd, gs):
                args += [ctypes.c_void_p(d.data_ptr()), ctypes.c_void_p(s.data_ptr()),
                         ctypes.c_long(_bytes_per_expert(s))]
            rc = h.expert_cache_gather(
                *args,
                ctypes.c_void_p(self.miss.data_ptr()),
                ctypes.c_void_p(self.n_miss.data_ptr()),
                self.chunks, self.lanes, stream,
            )
            if rc != 0:
                raise RuntimeError(f"expert_cache_gather failed rc={rc}")

        return self.remap(topk_ids)
