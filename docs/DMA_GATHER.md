# The copy-engine gather: measured, and it does not help

Superseded conclusion. This file previously argued that the gather kernel ran at
~5.8 GB/s and that switching to hipMemcpyAsync would take a GLM-5.3 decode step
from 128 ms to ~41 ms. Both halves were wrong. Kept as a record of how.

## What was actually measured, on 2x MI210 (gfx90a), GLM-5.3-Flash decode

| arm | ms/step | tok/s |
|---|---|---|
| kernel gather (default) | 130.4-132.6 | 7.54-7.67 |
| DMA gather (EXPERT_CACHE_DMA=1) | 134.1-137.1 | 7.29-7.46 |

The DMA path is slightly *slower*. And `tests/test_policy.py` on the same box
reports the existing kernel at **23-27 GB/s** at production shapes
(`gather 8 experts: 392.5 us, 26.7 GB/s`), i.e. the same rate an engine copy
achieves. There was never a 4.7x to win.

## Where the 5.8 GB/s came from

It was inferred, not measured: the step was 128.8 ms, compute was assumed to be
~16 ms, the remainder was assumed to be all transfer, and a bandwidth was divided
out of that assumption. The assumption was false.

Two later measurements falsify it directly:

- `EXPERT_CACHE_NOGATHER=1` (skip the copy entirely, accept wrong output, keep the
  timing) moved the step 128.8 -> 120.7 ms. The whole gather is worth ~8 ms, 6% of
  the step -- not 112 ms.
- CUDA events around `refresh()` report ~0.9 ms/call, which would be ~38 ms/step,
  far more than the 8 ms its removal actually saves. The gather overlaps other
  work, so an event window around it measures queue residency, not exclusive cost.
  Phase timings taken this way do not sum to the step and must not be treated as
  a budget.

## Why the DMA path is kept but defaulted off

`_gather_dma()` reads the device-side miss list back to the host, so it needs a
`stream.synchronize()` per layer. That breaks the invariant this module is built
on (`cache.py:13`, and README: "Graph capture is worth about 4x, so staying
capturable dominates every other design consideration"). It costs capturability
and buys nothing. `EXPERT_CACHE_DMA=1` remains only so the comparison can be
re-run; there is no configuration in which it is currently the right choice.

## What the manager costs

`expert_cache_manage` measures **11.6 us/call** in isolation on this box, matching
the 11.3 us in docs/results.md. An earlier claim in this branch that it cost
~420 us/call was an artefact of the same event-window mistake described above.
The kernel is fine.
