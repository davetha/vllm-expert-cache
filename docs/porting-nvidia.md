# Porting vllm-expert-cache to NVIDIA GPUs

A brief for someone with an NVIDIA card who wants to try this. Written by the people who built
the AMD version, who do **not** have an NVIDIA card to test on.

Repo: https://github.com/davetha/vllm-expert-cache

---

## Bottom line

**The port is close to free.** The interesting question is not whether it works — it is whether
it is worth anything on your particular host link. Read the "Is this worth it on your hardware"
section before writing code.

Two facts that make this cheap:

- **The Python layer needs no changes at all.** It already calls
  `torch.cuda.current_stream().cuda_stream` and allocates on `torch.device("cuda")`. Under ROCm
  those map to HIP, which is why it works today; on NVIDIA they are simply CUDA.
- **The kernels touch four HIP symbols.** Not four categories — four. Everything else in the
  file is syntax CUDA and HIP already share.

vLLM's CUDA backend is also its primary, best-tested platform, so unlike the Intel port there is
no real doubt that the foundation this sits on (`--cpu-offload-params experts`) works.

---

## What this package does

MoE models are mostly expert weights, and those weights are cold — each token routes to a handful
of experts while the rest occupy VRAM. vLLM can move experts to host memory with
`--cpu-offload-params experts`, freeing a lot of VRAM, but then every expert read crosses PCIe
and decode collapses.

This puts a small VRAM cache in front of those host-resident experts. Two device kernels run per
MoE layer per forward, ahead of the fused MoE GEMM: `expert_cache_manage` picks victims among
slots this step does not need and emits a miss list, `expert_cache_gather` copies the missing
experts host-to-slot. Routing ids are then remapped to slot space and the **stock** fused MoE
kernel runs unchanged. There is no custom GEMM.

On 2x AMD MI210 (PCIe 4.0 x16) with Qwen3-30B-A3B W8A8, 128 experts, top-8:

| Experts in VRAM | Cache | Decode | vs. no cache |
| --- | --- | --- | --- |
| 128 (no offload) | n/a | 65.7 tok/s | 3.4x |
| 64 | on | 54.3 tok/s | 2.8x |
| 32 | on | 39.4 tok/s | 2.1x |
| 0 (offloaded) | off | 19.1 tok/s | 1.0x |

---

## Is this worth it on your hardware? Ask first.

This package converts host-link bandwidth into throughput. Its value therefore scales inversely
with how fast your host link already is, and NVIDIA spans a much wider range than AMD does here.

- **Consumer / workstation cards on PCIe 4.0 or 5.0 x16** (RTX 4090, 5090, A6000, and similar).
  Roughly our regime — 25-60 GB/s host-to-device. Expect behaviour close to the numbers above.
  This is the sweet spot.
- **Grace Hopper / Grace Blackwell (GH200, GB200).** NVLink-C2C runs host memory at hundreds of
  GB/s, an order of magnitude past PCIe. Offloading experts there is *already cheap*, so the
  un-cached floor is far higher and there is much less for a cache to recover. **This package may
  be close to pointless on those systems, and that is the correct outcome** — the problem it
  solves has been solved in hardware.
- **Multi-GPU with NVLink between GPUs.** Irrelevant to this. NVLink between GPUs does not speed
  up host-to-device transfer, which is the path that matters here.

Cheap way to find out before porting anything: serve an MoE model with and without
`--cpu-offload-gb N --cpu-offload-params experts` and compare decode. **The gap you measure is
the entire budget this package can recover from.** If offload costs you 10%, the best possible
outcome is recovering some of that 10%. If it costs you 3.5x, as it did for us, there is a lot on
the table.

---

## The port

### 1. Kernels

Source: `kernels/expert_cache.hip` (~630 lines). The complete HIP-specific surface is:

| In the source | CUDA equivalent |
| --- | --- |
| `#include <hip/hip_runtime.h>` | `#include <cuda_runtime.h>` |
| `hipStream_t` | `cudaStream_t` |
| `hipGetLastError()` | `cudaGetLastError()` |
| `hipLaunchKernelGGL(k, grid, blk, shmem, stream, args...)` | `k<<<grid, blk, shmem, stream>>>(args...)` |

Plus one clang-ism, on line 30:

```c
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
```

This is the 16-byte copy unit for the gather. Replace with CUDA's native `uint4`, which is also
16 bytes and supports the assignment the gather does. `nvcc` will not accept `ext_vector_type`.

Everything else — `__global__`, `__shared__`, `__syncthreads()`, `threadIdx`, `blockIdx`,
`__launch_bounds__`, `__device__ __forceinline__`, `__restrict__` — is already common to both.
There are **no warp-width intrinsics anywhere**: no shuffles, no ballots, no assumption about 32
or 64 lanes. Every reduction is block-level through shared memory. So nothing has to be
restructured for warp size 32.

A shim header mapping those four symbols plus the typedef is cleaner than editing the body, and
keeps one source building for both vendors. HIP's own NVIDIA backend (`HIP_PLATFORM=nvidia`)
technically exists, but it routes through `nvcc`, which will still reject the vector typedef — so
the shim is the more predictable route.

### 2. Build

`kernels/build.sh` is AMD-specific in two flags: `--offload-arch=gfx*` becomes `-arch=sm_XX` (or
`-gencode`), and `--rocm-device-lib-path` is dropped entirely. The export check at the bottom is
worth keeping — it catches a build that silently loses an entry point, which otherwise surfaces
much later as a confusing dlopen failure.

Keep the exported names `expert_cache_manage` and `expert_cache_gather`; the Python side binds
them by name through ctypes. (`expert_cache_fused` exists but the plugin does not use it — skip
it initially.)

### 3. Python

Nothing. Genuinely nothing.

---

## Prove it is correct before measuring

```bash
POLICY=0 python tests/test_policy.py     # LRU
POLICY=1 python tests/test_policy.py     # LFU
```

Drives the real kernels with random and skewed routing and checks the expert-to-slot table, cold
map, slot contents, priorities and miss list against an **independent numpy implementation after
every step**, then verifies gathered bytes match their source rows. A subtly wrong port fails
here immediately and names the field that diverged. Do not benchmark until both pass.

Two contracts to preserve:

- **The manager must stay deterministic.** No atomics, no order-dependent reduction. With tensor
  parallelism every rank sees identical routing and must evolve an identical cache with no
  communication between them; introduce an atomic and ranks silently diverge.
- **The gather reads host memory from the device.** On CUDA this is the well-trodden pinned-host
  path that vLLM's offloader already sets up. Per-expert slabs must stay 16-byte multiples.

Then confirm the cache is actually engaged in a real server, because throughput alone will not
tell you:

```bash
EXPERT_CACHE_DISABLE=1 vllm serve ...    # vs. the same command without it
```

On AMD that is 19.1 against 54.3 tok/s. If disabling changes nothing, the plugin is not active —
usually a slot budget below `max_num_seqs x top_k`, or serving without
`--cpu-offload-params experts`.

---

## Measuring, without fooling yourself

We wasted hours here. Please don't repeat it.

**Count misses, not tokens.** `python tests/compare_policies.py` replays one routing trace under
each policy and counts experts fetched. Deterministic, no server needed, and it measures exactly
what the cache controls.

**If you use throughput, the unit of replication is the server process, not the request.** Nine
runs inside one loaded server agreed to 0.02 tok/s and were still not reproducible — restarting
moved the number 1.5 tok/s, fifty times that spread. We published two confident and opposite
conclusions before catching it. Discard the first run, take five more, repeat across restarts.

**Sizing trap.** A step is served from slots only when `tokens x top_k <= slots`. With
`--max-num-seqs 4` and top-8 routing a batched decode step carries 32 routing ids, so a budget
under 32 makes concurrent traffic bypass the cache silently while single-stream benchmarks still
look healthy.

---

## Sizing

Only the non-expert weights and the KV cache must stay permanently resident. For the model we
tested (~30 GB in W8A8) that split as **~27 GB of expert weights** (128 experts, ~212 MB each)
against **~3 GB of everything else**. So on a card with VRAM `V`, roughly `(V - 3 GB - KV)` is
available for slots, at ~212 MB per expert.

| Card | VRAM | Rough slot budget | Expected, from our curve |
| --- | --- | --- | --- |
| RTX 4090 / 3090 / A6000-class | 24 GB | ~80-90 of 128 | ~85-90% of fully resident |
| RTX 5090 | 32 GB | model fits outright | no need to offload |
| A100 | 40 GB | fits outright | no need to offload |

The honest implication: for *this* model a 24 GB card barely needs the cache and a 32 GB card
does not need it at all. This package earns its keep when experts genuinely do not fit — a much
larger MoE on a 24 GB card, or a mid-size MoE on 12-16 GB. Scale the model to the card, not the
other way round.

---

## What we know, and what we are guessing

Verified on AMD: builds for gfx90a, gfx942 and gfx1201; fully validated on gfx90a with reference
tests, serving, and the numbers quoted here; the kernel originated on RDNA4 and ran in production
there, so it has worked at both 32- and 64-wide execution in anger.

Unknown for NVIDIA:

1. Whether the mechanical port compiles and passes `tests/test_policy.py` first time. We expect
   yes; nobody has run it.
2. Whether the benefit survives on your host link. See the section above — this is the real
   question, and on Grace Hopper the honest answer may be "no, and that is fine."
3. ~~Whether your model uses the one supported quantisation path.~~ No longer a
   constraint: the plugin is quantisation-agnostic and discovers per-expert tensors by
   shape, so any `FusedMoEMethodBase` subclass is fair game. It refuses, loudly and with a
   reason, on monolithic methods, expert parallelism, non-16-byte-aligned per-expert slabs,
   and any scheme whose per-expert tensors it cannot fully rebind.

Two traps that cost us time writing the AMD backend and will recur in any new one:

- Read the source tensors fresh on every call. vLLM's offloader relocates them to host memory
  *after* `process_weights_after_loading`, so a pointer captured at setup dangles and the gather
  faults.
- Weights and their scales are indexed by the same id, so both must be slot-indexed. Scales
  reach the kernel via a `FusedMoEQuantConfig` whose fields are read-only properties and cannot
  be swapped per call. The generic backend handles this without rebuilding any kernel: it
  rebinds the layer's attributes to slot buffers, re-runs the scheme's own
  `get_fused_moe_quant_config(layer)`, and assigns the result to
  `moe_kernel.fused_experts.quant_config` for the duration of the call.

---

## Reporting back

Issues and results welcome at https://github.com/davetha/vllm-expert-cache — including a negative
result. "Offload was already cheap on this system so the cache bought nothing" is genuinely
useful and would save the next person the effort.
