# vllm-lru-expert-cache

A device-side **LRU cache for MoE experts** in [vLLM](https://github.com/vllm-project/vllm).

Mixture-of-experts models are mostly expert weights, and those weights are cold: each token
routes to a handful of experts, so the rest sit in VRAM doing nothing. vLLM can already push
experts to host memory with `--cpu-offload-params experts`, but then *every* expert read
crosses PCIe and decode throughput collapses.

This package keeps a bounded set of experts resident in VRAM and lets an LRU decide which
ones. Routing has strong temporal locality, so a fraction of the experts catches most of the
reads, and you get most of the fully-resident speed at a fraction of the expert VRAM.

Measured on 2x AMD Instinct MI210 (gfx90a), Qwen3-30B-A3B W8A8 (128 experts, top-8), TP2,
single-stream greedy decode:

| Experts kept in VRAM | Decode | vs. fully resident |
| --- | --- | --- |
| 128 (all resident) | 65.7 tok/s | 100% |
| 96 | 58.4 tok/s | 89% |
| 64 | 54.3 tok/s | 83% |
| 48 | 49.3 tok/s | 75% |
| 32 | 39.4 tok/s | 60% |
| 16 | 30.8 tok/s | 47% |
| 0 (offload, no cache) | 19.1 tok/s | 29% |

Half the experts resident holds **83% of full speed**; a quarter still holds **60%**, and
**2.1x** the un-cached offload floor.

## Install

```bash
git clone <this repo> && cd vllm-lru-expert-cache
kernels/build.sh gfx90a          # or gfx942, gfx1201, or several at once
pip install -e .
```

That is the whole setup. vLLM discovers the package through its `vllm.general_plugins`
entry point and arms the cache at engine startup — no launch flag, no code change, no
patched vLLM checkout.

Run the model with expert offload as usual; the cache layers on top of it:

```bash
LRU_CACHE_SLOTS=64 vllm serve <model> \
  --tensor-parallel-size 2 \
  --cpu-offload-gb 22 --cpu-offload-params experts
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `LRU_CACHE_SLOTS` | — | Experts kept resident, as an absolute count |
| `LRU_CACHE_FRACTION` | `0.5` | Used when `LRU_CACHE_SLOTS` is unset: fraction of experts to keep |
| `LRU_CACHE_DISABLE` | `0` | `1` turns the cache off without uninstalling |
| `LRU_CACHE_LIB` | — | Path to `liblruexpert.so` if it is not beside the package |
| `LRU_CACHE_POLICY` | `lfu` | Victim rule: `lfu` (frequency + decay) or `lru` (recency) |
| `LRU_CACHE_DECAY` | `64` | LFU only: halve counts every N steps |
| `LRU_CACHE_CHUNKS` / `LRU_CACHE_LANES` | `16` / `64` | Gather kernel grid shape |

## How it works

Two device kernels run once per MoE layer per forward, ahead of the fused MoE GEMM:

1. **`lru_manage`** — one workgroup. Marks the experts routed this step, refreshes their
   LRU stamps, picks victims for the missing ones (argmin over stamps, restricted to slots
   whose expert is *not* needed this step, so nothing in use is ever evicted), rewrites the
   expert-to-slot table, and emits a miss list.
2. **`lru_gather`** — copies each missing expert's rows from host memory into its new slot.

The routing ids are then remapped from expert space to slot space and the **stock** fused
MoE kernel runs against the slot buffers. No custom GEMM: the cache sits entirely in front
of whatever kernel the quantisation backend already selected.

Wide steps — prefill, or any batch touching more distinct experts than there are slots —
skip the cache and read through the host copies on the unmodified path. Correctness never
depends on the cache being warm.

### Why it is branch-free

Everything above is deliberately free of device-to-host synchronisation: the "does this
step fit in the slots?" test is `tokens x top_k <= slots`, which is shape metadata, not a
device value. That matters more than it sounds. A *static* hot set — pick the N most-used
experts, pin them, read through on a miss — needs a per-step "are all of these resident?"
check, and that host sync cannot be captured into a CUDA/HIP graph. On the same hardware a
static set is stuck in eager mode at ~14 tok/s regardless of budget, while this cache
captures cleanly and reaches 54. The LRU's advantage here is as much about staying
graph-capturable as about hit rate.

Both tensor-parallel ranks observe identical routing and the kernels use no atomics, so
every rank's cache evolves identically with no cross-rank communication.

## Does a smarter policy help?

Yes -- LFU fetches **11-26% fewer experts** than LRU across budgets and workloads. Measured by
counting inserts over an identical routing trace (`tests/compare_policies.py`), which is
deterministic and needs no server:

| Workload | 16 slots | 32 slots | 64 slots |
| --- | --- | --- | --- |
| single stream | -17.2% | -20.3% | -11.1% |
| 4 tasks interleaved | -17.0% | -18.6% | -25.8% |

End-to-end that is worth only 1-2% here (4 concurrent streams: LRU 50.6/50.7 vs LFU 51.3/51.5
tok/s across two serves), because PCIe transfers are a fraction of decode time at these budgets.
On a slower link, where each avoided transfer is worth more, expect the gap to widen.

**LFU is the default.** Both rules are now validated bit-exactly against the reference model
(`tests/test_policy.py` checks either policy against an independent numpy implementation on every
field of every step), so there is no verification argument left for keeping the weaker one. Set
`LRU_CACHE_POLICY=lru` to go back.

The decay period (`LRU_CACHE_DECAY`, default 64 steps) is untuned -- it was picked as a round
number and never swept. If you care about the last few percent, that is the knob to explore.

**Measure misses, not tokens.** Between-serve throughput variance here is 3-6% while the policy
effect is 1-2%, so timing tokens produces sign flips rather than answers -- it did exactly that to
us three times running. See `docs/results.md`.

## Supported backends

- compressed-tensors **W8A8 int8** MoE (`CompressedTensorsW8A8Int8MoEMethod`)

The cache mirrors whichever per-expert tensors a backend needs — for int8 that is the two
weight matrices and their per-channel scales, because the fused kernel indexes weights and
scales with the same id. Adding a backend means listing its per-expert tensors and pointing
its kernel at the slot buffers; see `vllm_lru_cache/backends/`.

## Hardware

Built and validated on CDNA2 (gfx90a). The kernels use only block-level LDS reductions and
`__syncthreads()` — no warp-width intrinsics — so the same source builds for wave64 (CDNA)
and wave32 (RDNA) without change. CUDA support needs the HIP kernels ported; the Python
side is device-agnostic.

## Tests

```bash
kernels/build.sh gfx90a
python tests/test_policy.py
```

Drives the real kernels with random and skewed routing and checks the table, cold map, slot
contents, stamps and miss list against an independent numpy LRU reference after *every*
step, then verifies the gathered bytes are identical to their source rows and times both
kernels. On an MI210 the manager costs ~11 us per layer-step and the gather runs at the
PCIe roofline (~27 GB/s).
