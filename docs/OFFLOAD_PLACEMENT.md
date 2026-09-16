# Which layers the offloader reaches, and why it matters

The cache can only help a layer whose weights actually left the GPU. That makes vLLM's
offload *placement* part of this plugin's problem surface, and there are two findings there:
one confirmed bug fix, one unconfirmed optimisation.

## 1. Arming a resident layer is worse than pointless (fixed)

vLLM's UVA offloader is whole-layer and budget-limited. It walks modules in construction
order and offloads each until `--cpu-offload-gb` is exhausted. If the budget does not cover
every MoE layer, the remainder stay **fully GPU-resident**.

The plugin used to arm those layers anyway. A resident layer with a slot cache:

* burns slot VRAM it can never miss into — measured 2.42 GiB/rank on a 42-layer model at
  8 slots with 25 of 42 layers resident;
* runs a manager and a gather every step that copy VRAM to itself.

**Detection is not obvious.** The UVA path leaves `.device` reading as the accelerator,
because an offloaded parameter becomes a device-addressable *view* of pinned host memory.
Checking the device tells you nothing. vLLM's own marker is the signal:

```python
getattr(param, "_vllm_is_uva_offloaded", False) or param.device.type == "cpu"
```

`arm()` now declines a layer whose weights never left the GPU, and logs it — because which
layers the offloader reached depends on your budget, so this is information the operator
needs rather than something to hide. Set `EXPERT_CACHE_DECLINE_RESIDENT=0` to restore the
old behaviour.

Freeing that VRAM is what makes a larger slot budget affordable on a constrained box.

## 2. WHICH layers get offloaded is chosen by accident (unconfirmed)

Construction order is not a property of the layer. And layers are not interchangeable —
replaying LFU per layer at a fixed budget on a real trace:

```
layer    requests   misses    hit %
L15          2917      326    88.8%     <- deep
L14          3473      763    78.0%
L08          3128     1228    60.7%
L01          2989     1610    46.1%     <- shallow
overall 65.0%, spread 42.7 pp
```

A clean gradient with depth: shallow layers route near-uniformly across experts (a cache
cannot help them), deep layers concentrate on a stable set (a cache serves most requests).
**A shallow layer on the host costs ~5x the PCIe traffic of a deep one.** The default order
offloads the shallow ones first, which is close to the worst available choice.

`EXPERT_CACHE_OFFLOAD_SKIP=<n>` holds the first n modules resident so the budget is spent
deeper instead.

**Status: NOT CONFIRMED.** Three interleaved restarts per arm, same 26 layers armed and same
45.36 GiB offloaded in both:

```
deep offloaded  per-restart means: 75.09  76.01  82.15   -> 77.75 ms
vLLM default    per-restart means: 80.35  80.59  80.30   -> 80.41 ms
+3.3%, 2 of 3 paired wins, distributions OVERLAP
```

+3.3% is below that machine's measurement floor. The per-layer hit-rate spread is solid; the
performance claim is not. An unexplained asymmetry is worth chasing: the default arm spans
0.29 ms across restarts while the deep arm spans 7.06 ms.

### Implementation warning

Do **not** reorder by materialising the generator:

```python
mods = list(modules_generator)     # WRONG - OOM
```

`modules_generator` *constructs* each layer as it is pulled; the stock list comprehension
builds one, offloads it (freeing its VRAM), then builds the next. Calling `list()` first
allocates the whole model on the GPU at once — 63.23 GiB then OOM partway through creating
expert weights. Placement must be decided per module as it arrives, and the returned list
must keep original positions, since the caller builds the layer stack from it positionally.

## Measuring this on your own model

```
GLM53_TRACE_DUMP=/path/route.json   # or equivalent hook on ExpertSlotCache.refresh
cache_sim.py route.json <slots>     # per-layer replay
```

If per-layer hit rate is flat, placement does not matter for you. If it spans tens of
points, the decision is worth making deliberately — but budget for a proper A/B, because
the effect did not clear the noise here.
