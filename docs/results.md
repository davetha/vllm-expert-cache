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

## Replacement policy: LRU vs LFU -- INCONCLUSIVE

The kernel supports a second victim rule (`LRU_CACHE_POLICY=lfu`), storing a per-slot hit count
with periodic halving instead of a recency stamp. **Whether it beats LRU is unresolved**, and the
measurements below are recorded mainly as a warning about how to measure this.

Each cell is one serve, 6-9 timed 200-token decodes, warm-up discarded, 16 slots:

| Condition | LRU | LFU | Apparent winner |
| --- | --- | --- | --- |
| serve 1 | 30.09 | 31.84 | LFU by 5.8% |
| serve 2 (identical config) | 31.20 | 30.13 | LRU by 3.6% |
| throttled gather (c=2, l=8) | 27.48 | 26.68 | LRU by 3.0% |

The sign is not stable. Re-running the *same* configuration moved LRU by 1.11 tok/s and LFU by
1.71 tok/s, so between-serve variance is 3-6% -- larger than any policy effect being claimed.
Pooled, LRU averages 30.65 and LFU 30.99, a gap well inside that noise. There is no evidence
either policy is better at this budget. LRU remains the default.

### The methodology trap

Within a single serve these measurements are extraordinarily tight: spreads of 0.02-0.15 tok/s
across nine runs. That precision is real but it answers the wrong question -- it is the
repeatability of one loaded process, not the reproducibility of a configuration. Restart the
server and the number moves by 50x that spread.

This burned two successive conclusions here. A two-run comparison said LFU was worse; a
nine-run-single-serve comparison said LFU was 5.8% better and called it decisive because the
ranges were disjoint. Both were artefacts. **The unit of replication has to be the serve.**

To actually resolve an effect this size, either run 5+ independent serves per arm and compare
serve means, or make the policy switchable at runtime so both arms can be interleaved inside one
process, which removes the between-serve term entirely. The latter is the cheaper experiment and
is not yet implemented.

The same caveat applies to the throttled-gather run, so it does **not** answer whether a slower
link would favour one policy: that comparison carries the identical confound.

### What this does not undermine

The slot-budget curve is unaffected. Differences between budgets are 5-10 tok/s -- several times
the ~1.5 tok/s serve noise -- so the shape is solid even though each individual point carries that
uncertainty. (Consistently, 32 slots measured 39.4 in one serve and 40.75 in another.) The same
holds for the cache-on vs cache-off control, which is a 20 tok/s effect.

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
