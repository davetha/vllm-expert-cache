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

The kernel supports a second victim rule (`LRU_CACHE_POLICY=lfu`), which stores a per-slot hit
count with periodic halving instead of a recency stamp. Measured at 32 slots — the tightest
budget, where thrashing is worst and any policy gain should be largest:

| Policy | Decode (tok/s), two runs |
| --- | --- |
| `lru` (default) | 39.4, 39.6 |
| `lfu` (decay 64) | 36.7, 39.0 |

Repeated at 16 slots, where each decode step needs 8 experts out of only 16 resident and
turnover is at its most severe:

| Policy | Decode (tok/s), two runs |
| --- | --- |
| `lru` (default) | 30.8, 30.7 |
| `lfu` (decay 64) | 30.3, 31.6 |

**LFU did not beat LRU at either budget.** At 16 slots the two are indistinguishable (LFU's
own run-to-run spread, 30.3-31.6, is wider than the gap between the policies), so the verdict
is not budget-dependent: tightening the cache does not create an opening for a smarter victim
rule. It measured the same at best and slightly worse on average, with
more run-to-run spread. Earlier simulation on a different model (DeepSeek-V4 routing traces) had
LFU edging LRU by ~0.2 points of miss rate, so this is workload-dependent rather than a general
result — but on Qwen3-30B-A3B there is nothing to gain by switching, and LRU stays the default.

Worth noting where the real headroom is: a miss costs ~50 us of PCIe for a ~1.3 MB expert, while
the whole manager kernel costs ~11 us. The policy decision is nearly free and the transfer is
not, so reducing *transfers* (prefetching the next layer's experts during the current layer's
compute) looks like a bigger lever than a better victim rule.

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
