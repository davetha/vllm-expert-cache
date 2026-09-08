# Porting vllm-expert-cache to Intel GPUs

A brief for someone with an Intel Arc / Data Center GPU who wants to try this. Written by the
people who built the AMD version, who do **not** have an Intel card to test on.

Repo: https://github.com/davetha/vllm-expert-cache

---

## What this package does

Mixture-of-experts models are mostly expert weights, and those weights are cold — each token
routes to a handful of experts while the rest occupy VRAM doing nothing. vLLM can already move
experts to host memory with `--cpu-offload-params experts`, which frees a lot of VRAM, but then
every expert read crosses PCIe and decode throughput collapses.

This package puts a small VRAM cache in front of those host-resident experts. Two device kernels
run per MoE layer per forward, ahead of the fused MoE GEMM:

1. `expert_cache_manage` — marks the experts routed this step, updates their priority, chooses
   victims among slots this step does *not* need, and emits a miss list.
2. `expert_cache_gather` — copies each missing expert's rows from host memory into its slot.

The routing ids are then remapped from expert space to slot space and the **stock** fused MoE
kernel runs against the slot buffers. There is no custom GEMM.

On 2x AMD MI210 with Qwen3-30B-A3B W8A8 (128 experts, top-8), keeping half the experts resident
gives 54.3 tok/s against 19.1 with offload and no cache — 2.8x — and 65.7 is the never-offloaded
ceiling. Full numbers in `docs/results.md`.

---

## Do these in order. Step 0 decides whether the rest is worth doing.

## Step 0 — Does vLLM's expert offload work on XPU at all?

**This gates everything and costs an afternoon at most. Do not skip it.**

This package does not offload anything itself. It caches experts that vLLM's own offloader has
already placed in host memory, in allocations the GPU can read directly. If that mechanism does
not work on XPU, there is nothing for this to sit on top of, and no amount of kernel porting
helps.

Test it with no reference to this package at all:

```bash
# 1. Serve any MoE model on XPU normally. Note decode tok/s.
vllm serve <moe-model> --max-model-len 8192

# 2. Serve it again with expert offload. Note decode tok/s and that it still produces
#    coherent output.
vllm serve <moe-model> --max-model-len 8192 \
  --cpu-offload-gb <enough to move the experts> --cpu-offload-params experts
```

Three outcomes:

- **It runs and is much slower** (we saw 65.7 → 19.1, about 3.5x). This is the good case: the
  offloader works and there is a large gap for a cache to recover. Continue.
- **It runs and is barely slower.** Either the experts did not actually move, or Intel's
  host-to-device path is fast enough that the problem this solves does not exist on your machine.
  Check how much VRAM was actually freed before concluding.
- **It errors, hangs, or produces garbage.** Stop. Fix or report that first — it is a vLLM XPU
  issue, not a problem with this package.

Relevant vLLM code: `vllm/model_executor/offloader/uva.py`. Note that it is visibly CUDA-centric
(pinned memory, cudagraph-oriented comments), so check specifically whether host allocations end
up readable by the device rather than merely resident on the host.

Also confirm your model actually uses a supported quantisation path (below), or the plugin will
correctly decline to engage and you will measure nothing.

---

## Step 1 — Fix this package's device assumptions

Small and mechanical, but it will not run without it. The Python layer is currently CUDA/ROCm
only in two places (under ROCm, `torch.cuda` transparently maps to HIP, which is why this went
unnoticed):

- `vllm_expert_cache/cache.py` — `torch.cuda.current_stream().cuda_stream`, the stream handle
  passed to both kernels.
- `vllm_expert_cache/backends/compressed_tensors_int8.py` — `torch.device("cuda", ...)` for the
  slot buffers.

Both need an accessor that resolves to `torch.xpu` on XPU. The kernels need whatever opaque queue
or stream handle your runtime uses in place of the HIP stream pointer.

---

## Step 2 — Port the kernels

Source: `kernels/expert_cache.hip` (~630 lines). Build: `kernels/build.sh`.

**This is easier than a typical cross-vendor port, for one specific reason: there are no
warp-width intrinsics.** No shuffles, no ballots, no sub-group size assumptions anywhere. That is
normally the hardest part of moving to Intel, because sub-groups are 8/16/32 wide against AMD's
32/64 and shuffle-based reductions have to be restructured. Here every reduction is already
block-level through shared memory and `__syncthreads()`. The same source already compiles for
both AMD wavefront widths (gfx90a and gfx942 at wave64, gfx1201 at wave32) unchanged, which is
good evidence it does not depend on execution width.

Mechanical mapping to SYCL:

| HIP | SYCL |
| --- | --- |
| `__global__` + `hipLaunchKernelGGL` | `parallel_for` over an `nd_range` |
| `__shared__ T x[N]` | `local_accessor<T>` |
| `__syncthreads()` | `nd_item.barrier(access::fence_space::local_space)` |
| `threadIdx.x`, `blockDim.x` | `nd_item.get_local_id(0)`, `get_local_range(0)` |
| `blockIdx.x` | `nd_item.get_group(0)` |
| `u32x4` (`ext_vector_type(4)`) | `sycl::vec<uint32_t, 4>` |

SYCLomatic will do most of it. Two contracts to preserve:

- **The manage kernel must stay deterministic.** It uses no atomics and no run-to-run ordering
  anywhere, by design: with tensor parallelism every rank sees identical routing and must evolve
  an identical cache with no communication. If you introduce an atomic or an
  order-dependent reduction, ranks will silently diverge.
- **The gather reads host memory from the device.** It needs Unified Shared Memory host
  allocations (`sycl::malloc_host` or equivalent) so the GPU can read expert rows over PCIe.
  Per-expert slabs must stay 16-byte multiples; the code assumes that.

Launch shapes: manage is a single workgroup of 256 threads. Gather is a 2D grid
(`EXPERT_CACHE_CHUNKS` x `EXPERT_CACHE_LANES`, default 16 x 64) of 256-thread blocks.

Exported entry points, which the Python side binds by name via ctypes — keep the signatures:
`expert_cache_manage`, `expert_cache_gather`. (`expert_cache_fused` exists but the plugin does
not use it; skip it initially.)

---

## Step 3 — Prove it is correct before you measure anything

```bash
POLICY=0 python tests/test_policy.py     # LRU
POLICY=1 python tests/test_policy.py     # LFU
```

This drives the real kernels with random and skewed routing and checks the expert-to-slot table,
the cold map, slot contents, priorities and the miss list against an **independent numpy
implementation after every single step**, then verifies gathered bytes are identical to their
source rows. If your port is subtly wrong, this catches it immediately and tells you which field
diverged. Do not proceed to benchmarking until both policies pass.

One subtlety if you touch the manager: it returns as soon as it knows there is nothing to insert,
so a step with no misses never reaches the aging pass. A reference model that decays
unconditionally drifts out of agreement within a few dozen steps.

Then confirm the cache is actually engaged in a real server — throughput alone will not tell you:

```bash
EXPERT_CACHE_DISABLE=1 vllm serve ...   # vs. the same command without it
```

On AMD that is the difference between 19.1 and 54.3 tok/s. If disabling changes nothing, the
plugin is not active. The usual causes are a slot budget below `max_num_seqs x top_k` (see below)
or serving without `--cpu-offload-params experts`.

---

## Step 4 — Measure, without fooling yourself

We wasted hours here; please don't repeat it.

**Count misses, not tokens.** `python tests/compare_policies.py` replays one routing trace under
each policy and counts how many experts were fetched. It is deterministic, needs no server, and
measures exactly what the cache controls. Throughput is a noisy proxy.

**If you must use throughput, the unit of replication is the server process, not the request.**
Nine runs inside one loaded server agreed to 0.02 tok/s and were still not reproducible —
restarting it moved the number by 1.5 tok/s, fifty times that spread. We drew two confident and
opposite conclusions from that before catching it. Discard the first run (the cache is still
filling), take at least five more, and repeat across restarts.

**Sizing trap.** A step is served from slots only when `tokens x top_k <= slots`. With
`--max-num-seqs 4` and top-8 routing, a batched decode step carries 32 routing ids, so a budget
below 32 makes concurrent traffic silently bypass the cache while single-stream benchmarks still
look fine. Size for the batch you actually serve.

---

## What we know, and what we are guessing

Verified on AMD:

- Builds for gfx90a (CDNA2), gfx942 (CDNA3) and gfx1201 (RDNA4), individually and as one binary.
- Fully validated on gfx90a: reference tests, serving, and the throughput numbers quoted here.
- The kernel originated on RDNA4 and ran in production there, so it has worked at both wavefront
  widths in anger, not just compiled.

Unknown for Intel, in the order that matters:

1. Whether vLLM's expert offload works on XPU with device-readable host memory. **This decides
   feasibility.**
2. Whether the compressed-tensors W8A8 int8 MoE path (`CompressedTensorsW8A8Int8MoEMethod`) is
   what XPU actually uses. This package patches that one method and nothing else; a different
   quantisation path needs a new backend file, which is about 100 lines — see
   `vllm_expert_cache/backends/`.
3. Whether Intel's host-to-device bandwidth makes the tradeoff worthwhile at all. On PCIe Gen4 we
   measured the gather running at 27 GB/s, the hardware roofline, and a miss costing ~50 us for a
   1.3 MB expert. If your link is slower, the cache should matter *more*, not less.

Two traps that cost us time when writing the AMD backend, likely to recur:

- Read the source tensors fresh on every call. vLLM's offloader relocates them to host memory
  *after* `process_weights_after_loading`, so a pointer captured at setup time dangles and the
  gather faults.
- Weights and their scales are indexed by the same id, so both must be slot-indexed. Scales reach
  the kernel through a `FusedMoEQuantConfig` whose fields are read-only properties and cannot be
  swapped per call; the int8 backend builds a second kernel once, bound to the slot scale buffers.

---

## Sizing for a small card (Arc B580 and similar)

A 12 GB consumer card is a *good* target for this, not a marginal one — the whole point is
running a model whose experts do not fit.

Worked example with the model we tested, Qwen3-30B-A3B W8A8 (~30 GB on disk). Splitting that as
measured: **~27 GB is expert weights** (128 experts, so ~212 MB each) and **~3 GB is everything
else** — attention, embeddings, norms. Only that ~3 GB plus the KV cache has to be permanently
resident. On a 12 GB card that plausibly leaves ~6-7 GB for expert slots, or **roughly 30 of the
128 experts**.

On our curve, 32 slots ran at 60% of fully-resident speed. So the realistic outcome is not "a bit
slower than a big GPU" — it is that a model which **cannot be loaded at all** on 12 GB becomes
runnable at around 60% of the speed of hardware with 30 GB of VRAM. That is the case this package
exists for.

Two B580 specifics:

- **PCIe 4.0 x8**, roughly half the host-to-device bandwidth of the x16 link we measured (27 GB/s).
  That makes each cache miss more expensive, which means the cache should matter **more** here,
  not less — the un-cached floor drops further than the cached figure does. It also makes the LFU
  policy (the default, ~11-26% fewer fetches than LRU) more valuable, since every avoided transfer
  is worth more.
- **Sub-group width.** Xe sub-groups are 8/16/32 against AMD's 32/64. This normally breaks ported
  kernels, and it is precisely why the absence of any shuffle or ballot in these kernels matters:
  there is no execution-width assumption to fix.

If 30B is too tight, any smaller MoE works the same way — the mechanism cares about the ratio of
expert weights to VRAM, not the absolute size. Start with whatever MoE your XPU stack already
serves correctly, since Step 0 has to pass on it first regardless.

These are estimates from our AMD measurements, not numbers from a B580. Treat them as a sizing
guide for the first experiment, not a prediction.

## Reporting back

Issues and results are welcome at https://github.com/davetha/vllm-expert-cache — including a
negative result on Step 0, which is genuinely useful information and would save the next person
the same afternoon.
