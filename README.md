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

| Experts in VRAM | Expert offload | Cache | Decode | vs. fully resident | vs. no cache |
| --- | --- | --- | --- | --- | --- |
| 128 (all) | off | n/a | 65.7 tok/s | 100% | 3.4x |
| 96 | on | **on** | 58.4 tok/s | 89% | **3.1x** |
| 64 | on | **on** | 54.3 tok/s | 83% | **2.8x** |
| 48 | on | **on** | 49.3 tok/s | 75% | **2.6x** |
| 32 | on | **on** | 39.4 tok/s | 60% | **2.1x** |
| 16 | on | **on** | 30.8 tok/s | 47% | **1.6x** |
| 0 | on | off | 19.1 tok/s | 29% | 1.0x (baseline) |

The last row is what expert offload costs you *without* this package, and the top row is the
speed you gave up to get the VRAM back. Every row between is the same offloaded model with the
cache turned on: same host-resident weights, same `--cpu-offload-params experts`, just a
different number of experts cached back into VRAM. At half the experts the cache recovers
**2.8x** the un-cached rate and lands within 17% of never having offloaded at all.

Returns diminish as the budget grows: the first 16 slots are worth more than the last 32.

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

Arming happens from vLLM's one generic post-load call site,
`model_loader.utils.process_weights_after_loading`, rather than by patching the ~38
`FusedMoEMethodBase` subclasses. That runs after every method has processed its weights and
built its kernel, and `apply` is then wrapped per *instance* — so no vLLM class is
monkeypatched, and nothing depends on which quantisation modules happen to be imported.
Note that `base_loader` imports that function by name, so the hook rebinds it in every
module already holding a reference as well as at the source; patching only the source
leaves the plugin silently inert.

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

Miss rate is the fraction of expert lookups that needed a PCIe fetch, over 600 steps of top-8
routing across 128 experts (4,800 lookups). Raw fetch counts in brackets:

| Workload | Slots | LRU miss rate | LFU miss rate | Fewer fetches |
| --- | --- | --- | --- | --- |
| single stream | 16 | 49.8% (2392) | 41.2% (1980) | **-17.2%** |
| | 32 | 32.4% (1555) | 25.8% (1239) | **-20.3%** |
| | 64 | 15.8% (758) | 14.0% (674) | **-11.1%** |
| 4 tasks interleaved | 16 | 47.7% (2289) | 39.6% (1899) | **-17.0%** |
| | 32 | 28.1% (1347) | 22.8% (1096) | **-18.6%** |
| | 64 | 10.2% (489) | 7.6% (363) | **-25.8%** |

Note how steeply the miss rate falls with budget — from ~50% at 16 slots to ~10-16% at 64. That
is the same curve the throughput table shows, measured directly and without any timing noise.
Reproduce with `python tests/compare_policies.py`.

End to end that is worth only 1-2% on this machine, because PCIe transfers are a fraction of
decode time at these budgets. On a slower link, where each avoided transfer costs more, expect
the gap to widen. Both policies are validated bit-exactly against the reference model.

---

## Supported backends

**Any of them, in principle** — the cache does not know what int8, fp8, mxfp4 or wNa16 mean,
and there is no per-scheme code to write. It relies on three conventions that every
`FusedMoEMethodBase` subclass in vLLM already follows:

1. `apply` reads the weights off the layer: `self.moe_kernel.apply(x, layer.w13_weight,
   layer.w2_weight, ...)`.
2. Per-expert scales, zero points and biases are reachable by re-running the method's own
   factory, `self.get_fused_moe_quant_config(layer)`, which every subclass implements by
   reading named attributes off `layer`.
3. The built kernel keeps that config in one mutable attribute,
   `self.moe_kernel.fused_experts.quant_config`.

So the cache discovers the per-expert tensors by shape (leading dim == expert count), hangs
slot buffers on the layer, re-runs the scheme's own config factory to get a slot-bound
config, and swaps both in for the duration of a cached step.

Validated end to end on two structurally different paths, with no code between them:

| Path | Method | Per-expert tensors discovered | Result |
| --- | --- | --- | --- |
| compressed-tensors W8A8 int8 | `CompressedTensorsW8A8Int8MoEMethod` | 4 (weights + channel scales) | 48/48 layers armed; **55.7 vs 18.7 tok/s** with the cache disabled (2.9x); greedy output byte-identical either way |
| unquantized bf16 | `UnquantizedFusedMoEMethod` | 2 (weights only, no scales) | 8/8 layers armed; greedy output byte-identical on 3 prompts either way |
| compressed-tensors W4A16 int4, group 128 | `CompressedTensorsWNA16MoEMethod` | 8 (packed weights + scales + g_idx + sort indices) | 8/8 layers armed; greedy output byte-identical on 3 prompts either way. Also the case that exercises >6 mirrored tensors, i.e. the multi-launch gather |

Two tensors, four, and eight; plain, and packed-and-aliased. Same code path for all three.

The byte-identical greedy output is the load-bearing check: the cache is meant to be
invisible to results and only visible in throughput.

The arming path — discovery, slot binding, and every refusal below — is additionally
covered for unquantized, int8-shaped and wNa16-shaped layers by `tests/test_generic.py`,
which runs on CPU and needs neither a GPU nor a served model.

### Where each scheme stands

Every row below was served twice on the same hardware -- once with the cache on, once with
`EXPERT_CACHE_DISABLE=1` -- and the greedy completions compared. **Identical output is the
assertion**; throughput is a separate question and only the int8 row was measured for it.
All of them run through the same code, and the "tensors" column is what discovery found
without being told anything about the scheme.

| Scheme | vLLM method | Registers | Tensors | Result |
| --- | --- | --- | --- | --- |
| unquantized bf16 | `UnquantizedFusedMoEMethod` | `w13_weight` | 2 | identical |
| compressed-tensors W8A8 int8 | `CompressedTensorsW8A8Int8MoEMethod` | `w13_weight` | 4 | identical, and **2.9x** on a real 30B model |
| compressed-tensors W8A8 fp8 | `CompressedTensorsW8A8Fp8MoEMethod` | `w13_weight` | 4 | identical |
| compressed-tensors W4A16 int4 | `CompressedTensorsWNA16MoEMethod` | `w13_weight_packed` + alias | 8 | identical |
| compressed-tensors W8A16 int8 (weight-only) | `CompressedTensorsWNA16MoEMethod` | `w13_weight_packed` + alias | 8 | identical |
| GPTQ int4 | `AutoGPTQMoEMethod` | `w13_qweight` + alias | 4 | identical |
| AWQ int4 | `AutoAWQMoEMethod` | `w13_qweight` + alias | 6 | identical |
| compressed-tensors MXFP4 | `CompressedTensorsW4A4Mxfp4MoEMethod` | `w13_weight` | -- | **not reached on ROCm**, for reasons upstream of this package -- see below |

Two, four, six and eight per-expert tensors; plain, packed, and packed-behind-an-alias;
scales, zero points, group indices. One code path, no per-scheme code.

#### Why MXFP4 does not load on ROCm

Nothing to do with the cache, and worth writing down because the failure is three
unrelated things stacking up:

1. `CompressedTensorsW4A4Mxfp4MoEMethod.__init__` picks its backend from four branches:
   `moe_backend == "b12x"` consults the backend oracle, CUTLASS is taken when the *device*
   supports it, XPU has its own, and **everything else falls through to Marlin**. ROCm
   matches none of the first three, so it always lands on Marlin.
2. `--moe-backend emulation` does not help: only the literal string `"b12x"` reaches the
   oracle, so every other value falls through the same `else`.
3. Marlin is a CUDA-only kernel family, so `torch.ops._C.gptq_marlin_repack` does not
   exist in a ROCm build. `process_weights_after_loading` raises `AttributeError` while
   repacking the experts -- before this package's hook runs at all.

Ironically MXFP4 would be the *easiest* case for the cache if it loaded: that method
rebinds `w13_weight_packed` to a plain `layer.w13_weight` and deletes the packed name, so
discovery would see the simple two-name shape rather than the packed-plus-alias one it
already handles. The blocker is upstream and NVIDIA-specific, not a cache limitation.

The remaining untested schemes -- NVFP4, ModelOpt's variants, gpt-oss MXFP4 -- register a
plain `w13_weight` or the same packed-plus-alias shape as the validated rows, so the
mechanism applies. That is a structural argument, not a measurement, and this table
deliberately keeps the two apart.

### What it declines, and why

Correctness never depends on the cache, so anything it cannot prove it can do correctly it
refuses out loud, logs the reason, and leaves the layer on the stock path:

| Refusal | Reason |
| --- | --- |
| Monolithic methods | `apply_monolithic` takes `router_logits` and routes internally, so there are no `topk_ids` to remap into slot space |
| Expert parallelism (`layer.expert_map` set) | routing ids are global and `expert_map` already folds them into a local range; composing that with the slot table is a second remap this does not implement |
| A per-expert tensor the config factory reaches but layer-binding cannot | it would stay bound to full `[E]` storage while the ids are already slot-space, so every row would be the wrong expert's — silently |
| A per-expert slab that is not a multiple of 16 bytes | the gather moves 16 bytes per lane and would drop the tail |

That third one is the important guard. It compares the rebuilt config against the full one
and names the field that leaked, so a scheme that hides a per-expert tensor produces a
refusal rather than quietly wrong output.

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

### Porting to another vendor

Three separate pieces, in increasing order of difficulty:

1. **The kernels.** Plain HIP with no matrix intrinsics and, more importantly, no warp-width
   intrinsics — no shuffles, no ballots, no sub-group assumptions. That is normally the hardest
   part of a cross-vendor port and here it does not exist, so HIP to CUDA is close to a rename
   and HIP to SYCL is a mechanical rewrite of the barrier and local-memory calls.
2. **This package's Python layer**, which is *not* currently device-agnostic: it resolves the
   stream via `torch.cuda.current_stream().cuda_stream` and allocates on `torch.device("cuda")`.
   Under ROCm those map to HIP, but another backend needs the accessor abstracted. Small, real,
   not yet done.
3. **The foundation underneath it.** This cache does not offload anything itself — it caches
   experts that vLLM's own `--cpu-offload-params experts` already placed in host memory. That
   offloader needs to work, with device-readable host allocations, on the target platform. If it
   does not, there is nothing for this to sit on top of, and no amount of kernel porting helps.

Point 3 is the one that decides feasibility, and it is a question about vLLM rather than about
this package.

Step-by-step briefs for the two vendor ports: [NVIDIA](docs/porting-nvidia.md) (nearly free —
four HIP symbols and one typedef; the Python layer needs no changes) and
[Intel / XPU](docs/porting-intel.md) (gated on whether vLLM's expert offload works there at all).

Note also that the limiting factor is the quantisation backend, not the GPU: this currently
wires up compressed-tensors W8A8 int8 and nothing else. See
[Supported backends](#supported-backends).

---

## Tests

```bash
kernels/build.sh gfx90a
POLICY=0 python tests/test_policy.py     # LRU kernels vs numpy reference
POLICY=1 python tests/test_policy.py     # LFU kernels vs numpy reference
python tests/compare_policies.py         # miss counts, policy vs policy
python tests/test_generic.py             # arming: discovery, binding, refusals (CPU only)
```

`test_generic.py` drives the quantisation-agnostic arming path against synthetic stand-ins
for vLLM's classes: that the right per-expert tensors are discovered for unquantized,
int8-shaped and wNa16-shaped layers, that the rebuilt quant config really ends up on slot
storage, that the layer is restored exactly as found, that `apply` still accepts vLLM's
keyword call, and that each of the four refusal conditions fires with a reason naming the
offending field.

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

- Quantisation-agnostic by construction, but only compressed-tensors int8 has been
  measured end to end; other schemes are exercised by the arming tests, not by a served
  model. Expect the packed int4 family to need the most scrutiny — Marlin-style repacking
  may not leave weights expert-major and 16-byte aligned, which the cache will refuse
  rather than mishandle.
- Expert parallelism and monolithic MoE methods are refused outright, not supported.
- The fit test is conservative. It bounds distinct experts by `tokens x top_k` rather than
  counting them, so batches that would have fit sometimes read through. A device-side count
  would need a host sync and cost graph capture, which is not worth it.
- `EXPERT_CACHE_DECAY` was picked as a round number and never swept.
- Prefill always reads through. Prefetching the next layer's experts during the current layer's
  compute is the obvious unexplored win: a miss costs ~50 us of PCIe for a ~1.3 MB expert while
  the whole manager kernel costs ~11 us, so cutting transfers beats reordering them.

## License

Apache-2.0.
