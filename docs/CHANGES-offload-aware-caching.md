# Branch: offload-aware-caching

Five changes, from a session optimising GLM-5.3-Flash (321B MoE, W4A16 int4, 182 GB) on
2x MI210 where the model exceeds VRAM by ~54 GB and expert weights cross PCIe Gen4 x16
every token. Decode 10.96 -> ~11.8 tok/s; prefill 2.1x-3.3x depending on prompt length.

Everything is behind a default-on environment gate and reverses without editing code.
**Read §6 before shipping: one default can break an existing user's working config.**

---

## 1. Decline to cache layers the offloader never moved — CONFIRMED FIX

`vllm_expert_cache/backends/generic.py`, `EXPERT_CACHE_DECLINE_RESIDENT` (default on)

vLLM's UVA offloader is whole-layer and budget-limited: it walks modules in construction
order and offloads until `--cpu-offload-gb` is exhausted. Any remainder stays fully
GPU-resident. The plugin armed those layers anyway, which is worse than pointless — the slot
buffers cost VRAM they can never miss into, and every step runs a manager and a gather that
copy VRAM to itself.

Measured on a 42-layer model at 8 slots with 25 of 42 layers resident: **2.42 GiB/rank of
dead slot buffers.**

Detection is not obvious. The UVA path leaves `.device` reading as the accelerator, because
an offloaded parameter becomes a device-addressable *view* of pinned host memory. Checking
the device tells you nothing; `_vllm_is_uva_offloaded` is the only reliable marker.

The decline is logged rather than silent — which layers the offloader reached depends on the
operator's budget, so it is information they need.

## 2. Stage experts into VRAM for wide steps — CONFIRMED, 3x PREFILL

`vllm_expert_cache/cache.py` (`WideScratch`), `generic.py`, `EXPERT_CACHE_WIDE_SCRATCH`

Prefill bypasses the cache on any step wider than the slot count (a 2048-token chunk at
top-8 is 16,384 routing entries), so the fused MoE GEMM read expert weights directly from
pinned host memory. That read is slow in a specific, measurable way: **~6.8 GB/s**, where
`expert_cache_gather_k` sustains **21-25 GB/s** on the same link. The GEMM tiles for HBM and
revisits; the gather streams contiguously, which is what PCIe wants.

Measured by varying offloaded-layer count on a fixed 2070-token prompt: 6,535 / 9,089 /
11,174 ms at 17 / 22 / 26 offloaded layers = **515 ms per offloaded layer**, 3.48 GiB moved.

Fix: stage the whole local expert set into VRAM with the gather kernel, then compute.

```
prompt tokens   before    after
   500           99.6     233.0
  2000          221.2     699.5
  7000          206.2     603.3
```

**Why it is safe:** the scratch is full-size and indexed BY LOCAL EXPERT ID exactly as the
source is, so `expert_map`, `global_num_experts` and the routing ids all keep their meaning.
Nothing is remapped — it is purely a change of where the weights live for the call. No kernel
change was needed: `expert_cache_gather_k` already reads its work list as (expert, slot)
pairs, so an identity list copies expert e to row e. One allocation is shared by all layers,
safe because layers run in order on one stream.

## 3. Fused EP remap — CORRECT, NO MEASURED GAIN

`vllm_expert_cache/fused_remap.py` (new), `EXPERT_CACHE_FUSED_REMAP`

The expert-parallel path ran four torch ops per layer per step on 8- and 288-element tensors
(cast + gather for local ids, index_select + masked_fill to compose the slot table). Now two
Triton kernels into persistent buffers.

Removes 126 kernels and ~1.13 ms of GPU busy time per step. **Wall clock did not move.** Under
CUDA graphs the Python body runs only at capture, so host-side savings are already free and
only GPU busy time counts — 1.13 ms of an 85 ms step is below the noise floor. Kept because it
is verified equivalent and does less work; not kept for speed.

## 4. Offload placement control — NOT CONFIRMED, DEFAULT NO-OP

`generic.py`, `EXPERT_CACHE_OFFLOAD_SKIP=<n>` (default 0 = untouched)

Per-layer LFU hit rate spans **46% to 89%** at a fixed budget — shallow layers route
near-uniformly, deep layers concentrate. A shallow layer on the host costs ~5x the PCIe
traffic of a deep one, and construction order offloads the shallow ones first.

Three interleaved restarts per arm: **+3.3%, 2 of 3 paired wins, distributions overlap** —
below the measurement floor. The hit-rate spread is solid; the performance claim is not.
See `docs/OFFLOAD_PLACEMENT.md`, including an unexplained variance asymmetry worth chasing.

**Implementation warning:** do not reorder by calling `list(modules_generator)`. It
*constructs* layers as pulled; materialising it allocates the whole model on the GPU at once
(63.23 GiB, then OOM). Decide per module as it arrives.

## 5. Docs

* `docs/POLICY_CEILING.md` — Belady's OPT: why the replacement-policy question is closed.
  Every online policy lands within ~1.5% of every other; the real distance is 12-16 pp to
  OPT. Includes the reuse-distance distribution that explains the gap, and the two cheap
  measurements that bound any future-use predictor before you build one.
* `docs/OFFLOAD_PLACEMENT.md` — the offloader interaction, both findings above.

---

## 6. Before shipping — decisions left open

**`EXPERT_CACHE_WIDE_SCRATCH` defaults to ON and allocates a full expert set** (1.74 GiB on
GLM-5.3-Flash). On the reference box that was paid for by dropping slots 56 -> 46. **For a
user already near their VRAM limit, upgrading would turn a working config into an OOM.** A
default that can break existing users is the wrong default for a shared library even when the
feature is good. Options: flip to opt-in, or probe free VRAM and decline with a log line.

**`tests/test_generic.py` is broken on this branch and was before these changes.** It
references `test_expert_parallel_is_refused` in its runner list at line 392; the function is
defined nowhere — a leftover from when EP support removed that refusal. Confirmed present in
the committed tree, `tests/` untouched here.

**Kernel tests pass**: `POLICY=0` and `POLICY=1` in `tests/test_policy.py`, both ALL CHECKS
PASSED, gather at 27 GB/s against the numpy reference.

**Measurement note.** The reference box moves **±0.9 tok/s between container restarts** —
the pinned offload buffer lands in different physical memory each time. Any claim below ~6 ms
needs interleaved restarts, 3+ per arm, arms alternated. Single-container comparisons there
are worthless, and two results in this session had to be retracted for exactly that.
