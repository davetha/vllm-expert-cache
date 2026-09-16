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



class WideScratch:
    """Full-size VRAM copies of one layer's expert weights, for steps too wide to cache.

    A prefill chunk routes to far more distinct experts than there are slots, so
    `fits()` is false and the layer used to read through to the host copies. That is
    correct but slow in a specific, measurable way: the fused MoE GEMM tiles for HBM and
    revisits, and pointed at a device-addressable view of pinned host memory it gets
    ~6.8 GB/s, against the 21-25 GB/s expert_cache_gather_k sustains on the same PCIe
    link doing wide contiguous copies (measured 2026-09-16: 515 ms per offloaded layer
    per 2070-token prefill, 3.48 GiB moved).

    So for a wide step, copy first and compute second. The scratch holds all E local
    experts and is indexed BY LOCAL EXPERT ID, exactly as the source tensor is, which is
    what makes this safe: expert_map, global_num_experts and the routing ids all keep
    their existing meaning and nothing is remapped. It is a pure change of where the
    weights live for the duration of the call.

    ONE allocation is shared by every layer of the same geometry, because layers run in
    order on one stream: layer N's GEMM is enqueued before layer N+1's fill, so the
    refill cannot overtake a read. At 144 experts x 12.38 MiB that is 1.74 GiB once
    instead of 1.74 GiB per layer.
    """

    _shared: dict = {}

    @classmethod
    def acquire(cls, cache, device):
        """The scratch for this layer's geometry, allocating it on first request."""
        key = (cache.E, tuple((n, tuple(cache.slots[n].shape[1:]), cache.slots[n].dtype)
                              for n in cache.names))
        inst = cls._shared.get(key)
        if inst is None:
            inst = cls(cache, device)
            cls._shared[key] = inst
        return inst

    def __init__(self, cache, device):
        self.names = list(cache.names)
        self.E = int(cache.E)
        self.buf = {
            n: torch.empty((self.E,) + tuple(cache.slots[n].shape[1:]),
                           dtype=cache.slots[n].dtype, device=device)
            for n in self.names
        }
        # The gather kernel reads its work list as (expert, slot) PAIRS
        # (expert_cache.hip: e = miss[2*j], sl = miss[2*j+1]). Repeating each index
        # twice gives [0,0, 1,1, 2,2, ...] -- the identity, so expert e lands at row e.
        self.pairs = torch.arange(self.E, dtype=torch.int32,
                                  device=device).repeat_interleave(2)
        self.n = torch.full((1,), self.E, dtype=torch.int32, device=device)
        self.bound: dict[str, torch.Tensor] = {}

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buf.values())

    def fill(self, owner, chunks: int, lanes: int) -> None:
        """Copy every local expert from wherever it lives into the scratch.

        Sources are read off `owner` at call time, not captured: the offloader moved
        these tensors to host memory after load, and the caller invokes this BEFORE
        rebinding the layer, so these are the originals rather than the scratch itself.
        """
        h = lib()
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        dsts = [self.buf[n] for n in self.names]
        srcs = [getattr(owner, n).data for n in self.names]

        # Same fixed-arity split-and-pad as the slot gather: the kernel copies exactly
        # GATHER_SLOTS buffers per launch, so a short final group repeats one it already
        # carries rather than passing unreadable memory.
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
                ctypes.c_void_p(self.pairs.data_ptr()),
                ctypes.c_void_p(self.n.data_ptr()),
                chunks, lanes, stream,
            )
            if rc != 0:
                raise RuntimeError(f"expert_cache_gather (wide scratch) failed rc={rc}")

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
        # Off by default: the copies themselves are ~4.7x faster than the gather
        # kernel, but this path has to read the device-side miss list back per
        # call, and that has not yet been shown to come out ahead end to end.
        # Set EXPERT_CACHE_DMA=1 to use it.
        self.dma_gather = os.environ.get("EXPERT_CACHE_DMA", "0") != "0"
        self._skip_gather = os.environ.get("EXPERT_CACHE_NOGATHER", "0") != "0"
        if self.dma_gather:
            self._h_miss = torch.empty_like(self.miss, device="cpu", pin_memory=True)
            self._h_nmiss = torch.empty_like(self.n_miss, device="cpu", pin_memory=True)

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

        if self._skip_gather:
            # DIAGNOSTIC ONLY. Runs the manager but copies nothing, so slots hold
            # whatever they held before: the output is deliberately wrong. The
            # point is the step time, which isolates what the copy actually costs
            # instead of inferring it from end-to-end timings.
            pass
        elif self.dma_gather:
            self._gather_dma()
        else:
            self._gather_kernel(h, stream)

        return self.remap(topk_ids)

    def _gather_dma(self) -> None:
        """Copy each missed expert with the copy engines instead of a kernel.

        expert_cache_gather_k moves a slab by having 256-thread blocks issue
        16-byte loads from the host-mapped source. That is latency-bound over
        PCIe and never reaches SDMA: measured ~5.8 GB/s against 24-27 GB/s for
        the same buffers under an engine copy (docs/DMA_GATHER.md).

        Tensor.copy_ takes the engine path even though the source is a UVA view
        whose .device reads as the accelerator -- the runtime knows the
        allocation is host-backed. Measured identical to a copy from the CPU
        tensor itself, 27.6 GB/s either way.

        The miss list lives on the device, so issuing per-copy from the host
        costs one read-back per call. That is the price of the mechanism swap
        and it is much smaller than what it buys.
        """
        # One sync, not two. Staging both buffers into pinned host memory and
        # synchronising once costs a single pipeline drain per call; reading
        # n_miss and then the list separately costs two, and on a 42-layer model
        # that difference is ~84 drains a step against ~42.
        self._h_nmiss.copy_(self.n_miss, non_blocking=True)
        self._h_miss.copy_(self.miss, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        n = int(self._h_nmiss[0])
        if n <= 0:
            return
        pairs = self._h_miss[: 2 * n].tolist()
        for name in self.names:
            dst = self.slots[name]
            src = getattr(self.owner, name).data
            for j in range(n):
                dst[pairs[2 * j + 1]].copy_(src[pairs[2 * j]], non_blocking=True)

    def _gather_kernel(self, h, stream) -> None:
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
