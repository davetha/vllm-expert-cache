# How much is left above LFU

The README shows LFU fetching 11-26% fewer experts than LRU. The natural next question is
whether something cleverer than either — ARC, LIRS, a learned future-use predictor — is
worth building. Belady's optimal answers that exactly, because it *is* a perfect predictor,
so the distance to it bounds every possible policy.

## Belady's OPT

From Bélády's 1966 paper on virtual-storage replacement. Also called MIN. One rule:

> On a miss, evict the item whose next use is furthest in the future.

Unimplementable live — it needs the future — but computable offline from a recorded trace.

**Why it is optimal**, by exchange: let A be optimal and find the first point it disagrees
with OPT. A evicts `x`, OPT evicts `y` (the furthest-future one). Build A' that evicts `y`
then imitates A. Since `y` is not needed for at least as long as `x`, A' matches A's later
choices with no more misses, and agrees with OPT one step longer. Repeat at each divergence:
A becomes OPT without the miss count ever rising.

The consequence: **the OPT gap is a bound, not an estimate.** No policy, model or heuristic
can close more of it than exists.

## Measured

GLM-5.3-Flash, 288 experts top-8 (144 EP-local per rank), 12,600 refreshes / 51,114 owned
requests over 17 armed layers, dumped with `GLM53_TRACE_DUMP` and replayed offline:

```
   S      LRU     LFU     ARC    LIRS    LRFU     OPT   best hit  OPT hit    gap
   8    37414   34153   36749   36488   34066   26969      33.4%    47.2%  13.9pp
  16    30993   28389   29794   30034   27814   19656      45.6%    61.5%  16.0pp
  32    22943   21354   22270   22209   20847   12531      59.2%    75.5%  16.3pp
  46    17819   16881   17451   17602   16512    9113      67.7%    82.2%  14.5pp
  64    12921   12313   12714   12784   12238    6268      76.1%    87.7%  11.6pp
  96     6710    6532    6654    6654    6541    3566      87.3%    93.0%   5.7pp
```

**Every online policy lands within ~1.5% of every other.** ARC and LIRS, both designed to
beat LRU on mixed workloads, do not beat plain LFU here. The interesting distance is not
between the policies; it is the 12-16 pp to OPT across the useful slot range.

Reproduce: `cache_sim.py <trace.json> <slots>`. OPT is O(n²) naively — sample long traces.

## Where the gap comes from

OPT's only advantage is knowing *when* an expert is next needed. So the gap is governed by
the reuse-distance distribution — requests between two uses of the same expert:

| percentile | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distance | 4 | 8 | 35 | 116 | 278 | 453 | 1007 | 3129 |

~30x spread between median and p99. **A frequency counter collapses that entire range into
one integer.** LFU cannot distinguish an expert due again in 4 requests from one due in
1,007 when their counts are similar, so it keeps both; OPT evicts the distant one and spends
the slot on something imminent. That difference is the whole gap.

This also explains why the decay/halving knob matters at all, and why it saturates: halving
is a crude way of letting old counts fade, i.e. a crude proxy for recency. It cannot recover
timing information that was never recorded.

## What it is worth end to end

On the reference machine (2x MI210, PCIe Gen4 x16, ~24 GB/s achieved), closing the *entire*
gap at S=46 over 26 armed layers:

```
misses/layer-step  1.3 -> 0.7     PCIe/step  ~360 MiB -> ~200 MiB
~160 MiB at 24 GB/s  =  ~6.5 ms of an ~80 ms step  =  ~8% ceiling
```

A real predictor captures a fraction of that. On a slower link, or with more armed layers,
the ceiling rises proportionally — this is the number to recompute for your own setup before
investing in policy work.

## If you are considering a future-use predictor

Two cheap measurements decide it, both from a routing trace:

1. **The OPT gap** (above) bounds anything that improves *eviction/admission*.
2. **Temporal locality** bounds anything that improves *prefetch*. Measured here:

```
21.3%  of a step's experts were also used in the previous step
 5.6%  of MISSES were experts used in the previous step
```

The misses are, nearly by definition, what recent history failed to anticipate. And routing
is decided layer-by-layer *during* a step, so there is no lead time: layer 30's requirement
is unknown until layer 30's router runs.

If your model shows materially higher persistence than 21%, prefetch may be worth more for
you than it is here.

## Belady's anomaly

Different result, same author: for FIFO, *increasing* cache size can *increase* misses. OPT
and LRU are immune, being stack algorithms — an n-slot cache's contents are always a subset
of an (n+1)-slot cache's. Relevant if a slot sweep ever produces a non-monotonic curve; with
LFU/LRU that indicates a measurement problem, not an anomaly.
