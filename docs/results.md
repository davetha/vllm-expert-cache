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

**LFU fetches 11-26% fewer experts than LRU.** Measured with `tests/compare_policies.py`, which
drives the real manage kernel over an identical routing trace under each policy and counts
inserts. Deterministic, no serving, so there is no throughput noise to hide behind:

| Workload | Slots | LRU misses | LFU misses | Change |
| --- | --- | --- | --- | --- |
| single stream, zipf | 16 | 2392 | 1980 | **-17.2%** |
| | 32 | 1555 | 1239 | **-20.3%** |
| | 64 | 758 | 674 | **-11.1%** |
| 4 tasks interleaved | 16 | 2289 | 1899 | **-17.0%** |
| | 32 | 1347 | 1096 | **-18.6%** |
| | 64 | 489 | 363 | **-25.8%** |

### Why that is only ~1-2% of throughput here

A miss costs one PCIe transfer, but transfers are a fraction of decode time at these budgets, so
cutting them by a fifth moves end-to-end throughput only slightly. Under 4 concurrent streams on
different tasks (32 slots, aggregate tok/s, plateau values, policies alternated across serves):

| Replicate | LRU | LFU |
| --- | --- | --- |
| A | 50.6 | 51.3 |
| B | 50.7 | 51.5 |

Both replicates favour LFU, by 1.3% and 1.6%.

**This is why the earlier throughput-only comparisons were worthless.** Between-serve variance is
3-6%; the real effect is 1-2%. Single-stream A/B runs flip-flopped (LFU +5.8%, then LRU +3.6%,
then LRU +3.0%) purely as noise around a small true value. Measuring a 1-2% effect with a 5% ruler
produces sign changes, not answers. **Count misses, do not time tokens** -- miss count is
deterministic, is what the policy actually controls, and needs no server at all.

### Expect a larger win on a slower link

The miss reduction is a property of the policy; the value of avoiding a miss scales with transfer
cost. On PCIe slower than the Gen4 x16 used here (~27 GB/s measured), the same 11-26% fewer
transfers converts into proportionally more throughput. These numbers are close to LFU's
weakest case.

### Recommendation

Prefer `LRU_CACHE_POLICY=lfu` when the budget is tight or the link is slow. LRU remains the
default only because it is the policy with a bit-exact reference test (`tests/test_policy.py`
validates policy 0 against a numpy model on every field of every step); LFU has no equivalent
yet. Writing that reference is the obvious next step before promoting LFU to the default.

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
