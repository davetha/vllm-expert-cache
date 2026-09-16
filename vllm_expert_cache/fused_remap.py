"""The two expert-id remaps under expert parallelism, one kernel each instead of four.

Per MoE layer per step the EP path in backends/generic.py ran four torch ops on tensors
of 8 and 288 elements: a cast and a gather to get local ids, then an index_select and a
masked_fill to compose the slot table into expert_map. At that size none of them is doing
measurable work -- a trace of one decode step showed the launch and dispatch, not the
copy. With 42 MoE layers that is 168 kernels per step to move ~1 KB.

Each remap is a gather with a guard, so each is one kernel:
  gather_local: local[i] = emap[ids[i]]                    (cast folded into the load)
  compose:      out[e]   = emap[e] < 0 ? -1 : table[emap[e]]

Both write into caller-owned persistent buffers so nothing allocates on the hot path and
the addresses stay stable across graph replays.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                     # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _gather_local_k(emap, ids, out, n, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        m = o < n
        e = tl.load(ids + o, mask=m, other=0).to(tl.int32)
        tl.store(out + o, tl.load(emap + e, mask=m, other=-1), mask=m)

    @triton.jit
    def _compose_k(table, emap, out, n, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        m = o < n
        loc = tl.load(emap + o, mask=m, other=-1).to(tl.int32)
        # An expert this rank does not own stays -1; clamping the index keeps the load
        # in bounds for those lanes, whose value is then discarded by the select.
        ok = m & (loc >= 0)
        slot = tl.load(table + tl.where(ok, loc, 0), mask=ok, other=-1)
        tl.store(out + o, tl.where(ok, slot, -1), mask=m)


def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1)).bit_length()


def gather_local(emap: torch.Tensor, ids: torch.Tensor,
                 buf: dict) -> torch.Tensor:
    """local ids for `ids`, i.e. emap[ids], into a buffer kept in `buf` per shape."""
    out = buf.get(ids.shape)
    if out is None:
        out = torch.empty(ids.shape, dtype=emap.dtype, device=emap.device)
        buf[ids.shape] = out
    n = ids.numel()
    if not HAVE_TRITON:
        return emap[ids.to(torch.int64)]
    _gather_local_k[(1,)](emap, ids.contiguous(), out, n, BLOCK=_pow2(n), num_warps=1)
    return out


def compose(table: torch.Tensor, emap: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Fold the slot table into expert_map: global -> slot, -1 where not owned."""
    n = emap.numel()
    if not HAVE_TRITON:
        torch.index_select(table, 0, emap.clamp(min=0).to(torch.int64), out=out)
        return out.masked_fill_(emap < 0, -1)
    _compose_k[(1,)](table, emap, out, n, BLOCK=_pow2(n), num_warps=4)
    return out
