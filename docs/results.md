# Measured results

## Setup

- **Hardware**: 2x AMD Instinct MI210 (gfx90a / CDNA2), PCIe Gen4, no XGMI bridge
- **Model**: Qwen3-30B-A3B-Instruct-2507, compressed-tensors W8A8 int8 (128 experts, top-8, 48 layers)
- **vLLM**: int8 fused-MoE path (`TritonInt8ScaledMMLinearKernel` + Triton int8 MoE), TP2,
  `--max-model-len 8192`, `--gpu-memory-utilization 0.85`, HIP graphs enabled
- **Offload**: `--cpu-offload-gb 22 --cpu-offload-params experts` (~13.6 GB of experts per worker
  moved to host memory)
- **Measurement**: single stream, greedy, 200-token completion, wall-clock; warmed with a
  40-token request first. Output checked for coherence at every configuration.

## Decode throughput vs resident-expert budget

| Experts in VRAM | Decode (tok/s) | vs resident |
| --- | --- | --- |
| 128 (no offload) | 65.7 | 100% |
| 96 | 58.4 | 89% |
| 64 | 54.3 | 83% |
| 48 | 49.3 | 75% |
| 32 | 39.4 | 60% |
| 16 | 30.8 | 47% |
| 0 (offload, cache disabled) | 19.1 | 29% |

Returns diminish as the budget grows: the first 32 slots roughly double throughput over the
un-cached floor, while the last 32 (96 -> 128) buy about 7 tok/s.

## Static hot set, for comparison

A static set of 64 experts (warm, then freeze, read through on a miss) measures **13.9 tok/s**
against the LRU's **13.7 tok/s** in eager mode — a tie, because eager per-layer dispatch cost
dominates and hides cache behaviour entirely.

The difference only appears with graphs enabled, and there the static design cannot compete:
its per-step residency check is a device-to-host sync that breaks graph capture, so it stays
near the eager number while the LRU captures and reaches 54.3.

## Kernel cost

From `tests/test_policy.py` on one MI210:

- `lru_manage`: ~11.3 us per layer-step at steady state
- `lru_gather`: 27.1 GB/s sustained — the PCIe Gen4 host-to-device roofline on this box
- Policy correctness: matches an independent numpy LRU reference on every field, every step,
  across five routing patterns; gathered bytes are identical to their source rows.

## Replacement policy: LRU vs LFU

The kernel supports a second victim rule (`LRU_CACHE_POLICY=lfu`), storing a per-slot hit count
with periodic halving instead of a recency stamp.

**The policy matters, but only when the cache is genuinely oversubscribed.** Nine runs per cell
in a single serve, first discarded as warm-up:

| Slots | LRU (mean, range) | LFU (mean, range) | LFU vs LRU |
| --- | --- | --- | --- |
| 32 | 40.75 (40.73-40.76) | 40.79 (40.40-40.95) | +0.1% -- a tie, ranges overlap |
| 16 | 30.09 (30.08-30.10) | **31.84** (31.78-31.93) | **+5.8% -- ranges do not overlap** |

At 32 slots the working set largely fits, both policies hold the same experts, and the choice is
irrelevant. At 16 the cache is genuinely oversubscribed and frequency wins: LRU will evict a
persistently-hot expert merely because it went untouched for a step, while LFU protects it.

### A methodology warning

An earlier two-samples-per-cell version of this comparison concluded the opposite -- that LFU was
no better and possibly worse. It was simply underpowered. Run-to-run scatter in the unsettled
first samples (~1.3 tok/s) swamped a real 1.75 tok/s effect and inverted the sign of the answer.
With nine runs the within-policy spread collapses to 0.02-0.15 tok/s and the separation at 16
slots is unambiguous.

Two samples cannot resolve a few-percent effect here. Discard the first run (the cache is still
filling: 28.1 vs a 30.1 steady state) and take at least five more.

## Verifying the cache is actually engaged

Set `LRU_CACHE_DISABLE=1` with everything else unchanged. On this setup that drops decode from
39.4 to 20.1 tok/s — the un-cached offload floor. Any measurement claiming a cache win should be
able to show this control.

## Locality is stronger than expected

16 slots is 1/8 of the experts, and each decode step routes 8 of them -- so on paper the cache
should turn over almost completely every step and collapse toward read-through. It does not:
30.8 tok/s is still **1.6x the un-cached floor** and 47% of fully resident.

That only works if consecutive tokens keep landing on overlapping experts. It is the same
property that makes the whole cache work, and it is why the victim rule barely matters: when
the reuse is that concentrated, recency and frequency identify nearly the same set.
