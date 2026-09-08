# vllm-expert-cache

Keep a bounded set of MoE experts resident in VRAM and stream the rest from host RAM, so a
mixture-of-experts model whose experts do not fit on the GPU still decodes at close to
fully-resident speed.

A vLLM plugin. Installing it is the entire setup — no launch flag, no patched vLLM checkout.

---

## The problem

MoE models are mostly expert weights, and those weights are cold: each token routes to a
handful of experts while the rest sit in VRAM doing nothing. vLLM can already move experts to
host memory with `--cpu-offload-params experts`, which frees a great deal of VRAM — but then
*every* expert read crosses PCIe and decode throughput collapses.

This package puts a small VRAM cache in front of those host-resident experts and lets a
device-side policy decide what stays. Routing has strong temporal locality, so a fraction of
the experts absorbs most of the reads.

Measured on 2x AMD Instinct MI210 (gfx90a), Qwen3-30B-A3B-Instruct-2507 W8A8 (128 experts,
top-8, 48 layers), TP2, single-stream greedy decode:

| Experts kept in VRAM | Decode | vs. fully resident |
| --- | --- | --- |
| 128 (all resident, no offload) | 65.7 tok/s | 100% |
| 96 | 58.4 tok/s | 89% |
| 64 | 54.3 tok/s | 83% |
| 48 | 49.3 tok/s | 75% |
| 32 | 39.4 tok/s | 60% |
| 16 | 30.8 tok/s | 47% |
| 0 (offload, cache disabled) | 19.1 tok/s | 29% |

Half the experts resident holds **83%** of full speed. Even an eighth still runs **1.6x** the
un-cached offload floor. Returns diminish as the budget grows: the first 32 slots roughly
double throughput, while the last 32 buy about 7 tok/s.

---

## Requirements

- vLLM with a supported MoE quantisation backend (see [Supported backends](#supported-backends))
- An AMD GPU and `hipcc`, from either a system ROCm install or the ROCm pip wheels that ship
  inside images like `rocm/vllm`
- A model served **with expert offload enabled** — the cache sits in front of the host copies
  that `--cpu-offload-params experts` creates, and does nothing without it

---

## Install

```bash
git clone <this repo> && cd vllm-expert-cache
kernels/build.sh gfx90a        # your arch: gfx90a, gfx942, gfx1201, or several at once
pip install -e .
```

`build.sh` finds `hipcc` automatically (system ROCm or pip wheels) and drops
`libexpertcache.so` next to the Python package. Verify it registered:

```bash
python -c "from importlib.metadata import entry_points; \
print([e.name for e in entry_points(group='vllm.general_plugins')])"
# -> ['expert_cache', ...]
```

vLLM loads the plugin at engine startup and patches the MoE layers itself.

---

## Usage

Serve the model with expert offload as usual. The cache layers on top:

```bash
EXPERT_CACHE_SLOTS=64 \
vllm serve <model> \
  --tensor-parallel-size 2 \
  --cpu-offload-gb 22 \
  --cpu-offload-params experts
```

That is the whole integration. `--cpu-offload-gb` decides how much leaves the GPU;
`EXPERT_CACHE_SLOTS` decides how much of it is cached back.

### Choosing a slot budget

Set it as high as your free VRAM allows — the curve above is monotonic, so more slots is
never slower. Two rules of thumb:

- **Prefer 50% of experts.** 83% of resident speed is usually the best trade.
- **Do not go below `max_num_seqs x top_k`.** This is the one that bites. A decode step is
  served from slots only when `tokens x top_k <= slots`; a wider step reads through instead.
  With `--max-num-seqs 4` and top-8 routing, a batched step carries up to 32 routing ids, so
  fewer than 32 slots means concurrent traffic silently bypasses the cache entirely. Size the
  budget for the batch you actually serve, not for single-stream benchmarks.

### Configuration

All configuration is environment variables. They deliberately avoid the `VLLM_` prefix, which
vLLM validates and warns about.

| Variable | Default | Meaning |
| --- | --- | --- |
| `EXPERT_CACHE_SLOTS` | — | Experts kept resident, absolute count |
| `EXPERT_CACHE_FRACTION` | `0.5` | Used when `SLOTS` is unset: fraction of experts to keep |
| `EXPERT_CACHE_POLICY` | `lfu` | Victim rule: `lfu` (frequency + decay) or `lru` (recency) |
| `EXPERT_CACHE_DECAY` | `64` | LFU only: halve counts every N steps. Untuned |
| `EXPERT_CACHE_DISABLE` | `0` | `1` disables the cache without uninstalling |
| `EXPERT_CACHE_LIB` | — | Path to `libexpertcache.so` if not beside the package |
| `EXPERT_CACHE_CHUNKS` / `_LANES` | `16` / `64` | Gather kernel grid shape |

### Confirming it is actually working

Throughput alone will not tell you — set `EXPERT_CACHE_DISABLE=1` and compare. Everything else
held equal, that drops decode from 54.3 to 19.1 tok/s here. If disabling changes nothing, the
cache was never engaged; the usual cause is a slot budget below `max_num_seqs x top_k`, or
serving without `--cpu-offload-params experts`.

---

## How it works

Two device kernels run once per MoE layer per forward, ahead of the fused MoE GEMM:

1. **`expert_cache_manage`** — one workgroup. Marks the experts routed this step, updates their
   priority, picks victims for the missing ones (argmin over priority, restricted to slots whose
   expert is *not* needed this step, so nothing in use is ever evicted), rewrites the
   expert-to-slot table, and emits a miss list.
2. **`expert_cache_gather`** — copies each missing expert's rows from host memory into its new
   slot.

The routing ids are then remapped from expert space to slot space and the **stock** fused MoE
kernel runs against the slot buffers. There is no custom GEMM: the cache sits entirely in front
of whatever kernel the quantisation backend already selected.

Wide steps — prefill, or any batch touching more distinct experts than there are slots — skip
the cache and read through the host copies on the unmodified path. Correctness never depends on
the cache being warm, so a miss is a slowdown and never a wrong answer.

### Why it is branch-free

The whole path avoids device-to-host synchronisation. The "does this step fit?" test is
`tokens x top_k <= slots`, which is shape metadata rather than a device value.

That matters more than it sounds. A *static* hot set — pick the N most-used experts, pin them,
read through on a miss — needs a per-step "are all of these resident?" check, and that host sync
cannot be captured into a CUDA/HIP graph. Measured here, a static set is stuck in eager mode at
~14 tok/s regardless of budget while this cache captures cleanly and reaches 54. Graph capture
is worth about 4x, so staying capturable dominates every other design consideration.

Both tensor-parallel ranks observe identical routing and the kernels use no atomics, so each
rank's cache evolves identically with no cross-rank communication.

### Replacement policy

`lfu` (default) stores a per-slot hit count with periodic halving; `lru` stores a recency stamp.
LFU fetches **11-26% fewer experts** across budgets and workloads:

| Workload | 16 slots | 32 slots | 64 slots |
| --- | --- | --- | --- |
| single stream | -17.2% | -20.3% | -11.1% |
| 4 tasks interleaved | -17.0% | -18.6% | -25.8% |

End to end that is worth only 1-2% on this machine, because PCIe transfers are a fraction of
decode time at these budgets. On a slower link, where each avoided transfer costs more, expect
the gap to widen. Both policies are validated bit-exactly against the reference model.

---

## Supported backends

- compressed-tensors **W8A8 int8** MoE (`CompressedTensorsW8A8Int8MoEMethod`)

Anything else is left untouched: the plugin logs that it found no supported backend and stays
out of the way.

### Adding one

A backend needs to declare which per-expert tensors to mirror and point its kernel at the slot
buffers. For int8 that is four tensors — the two weight matrices and their per-channel scales,
because the fused kernel indexes weights and scales with the same id. See
`vllm_expert_cache/backends/compressed_tensors_int8.py`; it is about 100 lines.

Two traps worth knowing before you write one:

- **Read the source tensors fresh on every call.** The expert offloader relocates them to host
  memory *after* `process_weights_after_loading`, so a pointer captured at setup time dangles
  and the gather faults.
- **Scales usually cannot be swapped per call.** They reach the kernel through a
  `FusedMoEQuantConfig` whose fields are read-only properties, so the int8 backend builds a
  second kernel once, bound to the slot scale buffers.

---

## Hardware

Any AMD GPU with ROCm. The kernels use only block-level LDS reductions and `__syncthreads()` —
no warp-width intrinsics — so one source serves both wavefront widths:

| Target | | Status |
| --- | --- | --- |
| `gfx90a` | CDNA2, MI210/MI250 | builds; full test suite and serving validated here |
| `gfx1201` | RDNA4, R9700 | builds; this kernel's original home, run in production there |
| `gfx942` | CDNA3, MI300 | builds; not yet exercised on hardware |

```bash
kernels/build.sh gfx90a gfx942 gfx1201    # one fat binary for all three
```

Nothing in the Python layer is device-specific. NVIDIA support needs the HIP kernels ported —
they are plain HIP with no AMD matrix intrinsics, so the port is mechanical, but it has not been
done.

Note that the limiting factor is the quantisation backend, not the GPU: this currently wires up
compressed-tensors W8A8 int8 and nothing else. See [Supported backends](#supported-backends).

---

## Tests

```bash
kernels/build.sh gfx90a
POLICY=0 python tests/test_policy.py     # LRU
POLICY=1 python tests/test_policy.py     # LFU
python tests/compare_policies.py         # miss counts, policy vs policy
```

`test_policy.py` drives the real kernels with random and skewed routing and checks the table,
cold map, slot contents, priorities and miss list against an independent numpy implementation
after *every* step, then verifies gathered bytes are identical to their source rows and times
both kernels. On an MI210: manager ~11.5 us per layer-step, gather 27 GB/s — the PCIe Gen4
host-to-device roofline, meaning the gather is running at hardware speed.

`compare_policies.py` replays one routing trace under each policy and counts inserts.

### If you benchmark this, count misses rather than timing tokens

Between-serve throughput variance here is 3-6%, while the policy effect is 1-2%. Timing tokens
to resolve that produces sign flips, not answers — it did exactly that to us three times, in
both directions, before we switched instruments. Miss count is deterministic, needs no server,
and is what the policy actually controls.

If you must use throughput: discard the first run (the cache is still filling — 28.1 against a
30.1 steady state), take at least five more, and treat the **serve** as the unit of replication.
Nine runs inside one server process agree to 0.02 tok/s and are still not reproducible; restart
it and the number moves fifty times that.

---

## Limitations

- One quantisation backend so far (compressed-tensors int8).
- The fit test is conservative. It bounds distinct experts by `tokens x top_k` rather than
  counting them, so batches that would have fit sometimes read through. A device-side count
  would need a host sync and cost graph capture, which is not worth it.
- `EXPERT_CACHE_DECAY` was picked as a round number and never swept.
- Prefill always reads through. Prefetching the next layer's experts during the current layer's
  compute is the obvious unexplored win: a miss costs ~50 us of PCIe for a ~1.3 MB expert while
  the whole manager kernel costs ~11 us, so cutting transfers beats reordering them.

## License

Apache-2.0.
