# The gather should use the copy engines, not a compute kernel

Measured on 2x MI210 (gfx90a) while running GLM-5.3-Flash, 2026-09-14.

`expert_cache_gather_k` copies a missed expert by having 256-thread blocks issue
16-byte loads from the host-mapped source. That is a compute kernel reading over
PCIe: it is latency-bound and never touches the SDMA copy engines. The
alternative, `hipMemcpyAsync` from the same buffers, is what the engines are for.

Same bytes, same source, measured with `torch.Tensor.copy_` on this hardware:

| transfer                      | pinned DMA   | pageable    |
|-------------------------------|--------------|-------------|
| w2 packed, 4.2 MB             | 25.51 GB/s   | 16.52 GB/s  |
| w13 packed, 8.4 MB            | 24.38 GB/s   | 17.55 GB/s  |
| one expert, 12.6 MB           | 26.84 GB/s   | 17.95 GB/s  |
| 64 MB bulk                    | 27.49 GB/s   | 18.41 GB/s  |
| 55 expert copies (one step)   | 27.63 GB/s, 25.1 ms      |

The in-engine gather achieves **~5.8 GB/s**. On GLM-5.3 at 32 slots that is
~650 MB of misses per token per rank taking ~112 ms of a 128 ms decode step.
At DMA rates the same traffic is ~25 ms, which would take the step to ~41 ms
(7.8 -> ~24 tok/s).

DMA reaches full rate at single-expert sizes -- 0.16 ms for 4 MB, 0.47 ms for
12.6 MB -- so per-copy launch overhead is not a reason to keep the kernel.

## Why the existing tunables cannot close this

`EXPERT_CACHE_CHUNKS` / `EXPERT_CACHE_LANES` set the gather grid. Raising chunks
16 -> 256 (so ~4 misses per layer occupy 1024 blocks instead of 64) moved a GLM
decode step 132.8 -> 128.8 ms, about 3%. The copy is not occupancy-limited; it is
limited by the mechanism. More threads waiting on PCIe latency is not more
bandwidth.

## Shape of the change

In `ExpertSlotCache.refresh`, replace the `expert_cache_gather` launches with one
`hipMemcpyAsync` per (missed expert x mirrored tensor) on a copy stream, then one
sync before the slot-bound call. The manager already produces the miss list and
slot assignments, so the bookkeeping is unchanged -- only the copy mechanism
moves. Two things to watch:

- the miss list is device-side, so issuing per-copy from the host needs it read
  back (a sync) or a device-side enqueue; a sync per layer costs ~8 ms/step on a
  42-layer model, which would eat a third of the win.
- the small mirrored tensors (scales, `g_idx`, sort indices) are launch-overhead
  dominated. Batch them or leave them on the kernel path.

This is not GLM-specific. Any deployment using the cache with
`--cpu-offload-params experts` pays the same rate on every miss; the win scales
with miss traffic. Fully-resident models see nothing, correctly.
