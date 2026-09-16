"""Quantisation-agnostic expert slot cache.

One backend covers every MoE quantisation scheme, because vLLM's `FusedMoEMethodBase`
contract is uniform in the three places that matter:

  1. Weights are read off the layer inside `apply`:
         self.moe_kernel.apply(x, layer.w13_weight, layer.w2_weight, ...)
  2. Per-expert scales, zero-points and biases are reachable by re-running the method's
     own factory, which every subclass implements by reading named attributes off `layer`:
         self.moe_quant_config = self.get_fused_moe_quant_config(layer)
  3. The built kernel keeps that config in exactly one mutable attribute:
         self.moe_kernel.fused_experts.quant_config

So the cache hangs slot buffers on `layer`, re-runs the method's own config factory to get
a slot-bound config, and swaps both in for the duration of a cached step. Nothing in this
file knows what int8, fp8, mxfp4 or wNa16 mean, and no per-scheme kernel is rebuilt.

Arming happens from vLLM's single generic post-load call site rather than by patching the
~38 `FusedMoEMethodBase` subclasses: `vllm.model_executor.model_loader.utils.
process_weights_after_loading` runs once, after every method has processed its weights and
built its kernel. `apply` is then wrapped per *instance*, so no class is monkeypatched at
all and nothing depends on which quantisation modules happen to be imported.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import sys

import torch

from .. import fused_remap

_FUSED_REMAP = os.environ.get("EXPERT_CACHE_FUSED_REMAP", "1") != "0"
from ..cache import ExpertSlotCache, WideScratch
from ..config import settings

# Fields on a FusedMoEQuantDesc that can hold a per-expert tensor.
_DESC_FIELDS = ("scale", "alpha_or_gscale", "zp", "bias")
_DESC_NAMES = ("_a1", "_a2", "_w1", "_w2")

_stats_every = int(os.environ.get("EXPERT_CACHE_STATS", "0"))
_installed = False


def _set_tensor(layer, name: str, t: torch.Tensor) -> None:
    """Rebind `layer.<name>`, respecting nn.Parameter registration.

    Assigning a bare Tensor over a registered Parameter raises, so parameters are
    rebound through `_parameters` with an already-wrapped Parameter.
    """
    if name in layer._parameters:
        layer._parameters[name] = t
    elif name in layer._buffers:
        layer._buffers[name] = t
    else:
        setattr(layer, name, t)


def _per_expert_tensors(layer, num_experts: int) -> dict[str, list[str]]:
    """Per-expert storages, each mapped to every name the layer reaches it by.

    A tensor indexed by expert id is exactly a tensor whose dim 0 is `num_experts`, so
    this finds the per-expert set without knowing the quantisation scheme. Mirroring one
    that turns out not to be expert-indexed would only waste VRAM -- it reaches a kernel
    only if the scheme's own config factory picks it up, and a factory that reads it does
    so by expert id. The dangerous direction is the opposite one (a per-expert tensor we
    fail to mirror), which `_leaked_full_tensors` catches below.

    Two subtleties, both learned from the packed compressed-tensors family:

    * Plain instance attributes count, not just registered parameters. wNa16/AWQ/GPTQ
      register `w13_weight_packed` (or `w13_qweight`) and then alias it with a bare
      `layer.w13_weight = layer.w13_qweight`, and `apply` reads the *alias*. Mirroring
      only the registered name would leave `apply` on full storage while the routing ids
      had already moved to slot space.
    * Names are grouped by storage. Those aliases are the same tensor under two names, so
      allocating a slot buffer per name would double the VRAM and, worse, let the two
      names disagree about which copy is current. One buffer per storage; rebind every
      name that points at it.
    """
    groups: dict[int, list[str]] = {}
    tensors: dict[int, torch.Tensor] = {}
    # remove_duplicate=False is load-bearing: torch dedupes by object identity, so an
    # aliased weight (`layer.w13_weight = layer.w13_qweight`, where the right-hand side
    # is a Parameter, so Module.__setattr__ registers it under both names) is otherwise
    # yielded under only ONE of its names -- and which one is arbitrary.
    for name, t in itertools.chain(
        layer.named_parameters(recurse=False, remove_duplicate=False),
        layer.named_buffers(recurse=False, remove_duplicate=False),
        ((k, v) for k, v in vars(layer).items() if isinstance(v, torch.Tensor)),
    ):
        if not (isinstance(t, torch.Tensor) and t.dim() >= 1 and t.size(0) == num_experts):
            continue
        key = t.data_ptr()
        if name not in groups.setdefault(key, []):
            groups[key].append(name)
        tensors.setdefault(key, t)
    # Key each group by its first name; that is the one the cache reads sources from.
    return {names[0]: names for names in groups.values()}


def _leaked_full_tensors(cfg, num_experts: int) -> list[str]:
    """Per-expert tensors still bound to full [E] storage after slot binding.

    This is the silent-corruption guard. If a scheme keeps a per-expert tensor somewhere
    `_per_expert_tensors` cannot see it -- a transposed layout, a sub-module, a value the
    factory computes rather than reads -- the rebuilt config would hand the kernel full
    expert-space rows while the routing ids have already been remapped into slot space.
    Every row would then be the wrong expert's, quietly. Refusing to arm the layer is the
    only safe answer, so report what leaked rather than guessing.
    """
    leaked = []
    for dname in _DESC_NAMES:
        desc = getattr(cfg, dname, None)
        if desc is None:
            continue
        for field in _DESC_FIELDS:
            t = getattr(desc, field, None)
            if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.size(0) == num_experts:
                leaked.append(f"{dname}.{field}{tuple(t.shape)}")
    return leaked


@contextlib.contextmanager
def _bound_to(layer, mapping):
    """Point every mirrored attribute at the given replacement for the duration."""
    saved = {name: getattr(layer, name) for name in mapping}
    try:
        for name, buf in mapping.items():
            _set_tensor(layer, name, buf)
        yield
    finally:
        for name, t in saved.items():
            _set_tensor(layer, name, t)


def _slot_bound(layer, cache: ExpertSlotCache):
    """Point every mirrored attribute at its slot buffer for the duration of the block."""
    return _bound_to(layer, cache.bound)


_STATS = {"n": 0, "miss": 0, "req": 0}


def _report_stats(cache, owned_ids, logger) -> None:
    """Log the running hit rate. Reads device counters, so it syncs -- diagnostic
    only, and silent unless EXPERT_CACHE_STATS is set to a reporting interval."""
    _STATS["n"] += 1
    if _STATS["n"] % _stats_every:
        # Only the sampled call pays the read-back. Counting every call would put
        # two device syncs in the hot path and change what is being measured.
        return
    _STATS["miss"] += int(cache.n_miss.item())
    _STATS["req"] += int((owned_ids >= 0).sum().item())
    if True:
        logger.info(
            "expert-cache stats: %d layer-steps, %d routed experts, %d misses "
            "-> hit rate %.1f%%",
            _STATS["n"], _STATS["req"], _STATS["miss"],
            100.0 * (1.0 - _STATS["miss"] / max(1, _STATS["req"])))


_TIMING = {"n": 0, "refresh": 0.0, "moe": 0.0, "remap": 0.0, "pend": [], "pend_remap": []}
_timing_every = int(os.environ.get("EXPERT_CACHE_TIMING", "0"))


def _drain_timing(logger) -> None:
    """Attribute the layer's time between refresh (manage + gather) and the MoE
    itself. Events are recorded per call and only read in bulk, so the hot path
    pays two event records rather than a sync."""
    for e0, e1, e2 in _TIMING["pend"]:
        e2.synchronize()
        _TIMING["refresh"] += e0.elapsed_time(e1)
        _TIMING["moe"] += e1.elapsed_time(e2)
    _TIMING["pend"].clear()
    for e0, eref, e1 in _TIMING["pend_remap"]:
        e1.synchronize()
        _TIMING["remap"] += eref.elapsed_time(e1)
    _TIMING["pend_remap"].clear()
    n = _TIMING["n"]
    if n:
        logger.info(
            "expert-cache timing: %d layer-calls | refresh %.3f ms/call | "
            "moe %.3f ms/call | remap %.3f ms/call",
            n, _TIMING["refresh"] / n, _TIMING["moe"] / n, _TIMING["remap"] / n)


def _decline(layer, method, num_experts: int) -> str | None:
    """Reasons this layer cannot be cached correctly. None means it can."""
    if method.is_monolithic:
        # apply_monolithic() takes router_logits and routes internally, so there are no
        # topk_ids to remap before the call and no way to reach slot space.
        return "monolithic method (routing happens inside apply_monolithic)"
    if getattr(method, "moe_kernel", None) is None:
        return "method has no moe_kernel (legacy non-modular path)"
    if getattr(method.moe_kernel, "fused_experts", None) is None:
        return "moe_kernel exposes no fused_experts to rebind"
    if getattr(method.moe_kernel.fused_experts, "consumes_expert_mask", False):
        # Under EP this kernel reads the 0/1 expert_mask rather than the
        # -1/local-id expert_map, so the slot table cannot be folded into it the
        # way apply() does below.
        return "kernel consumes expert_mask (slot composition not implemented)"
    if settings.slots_for(num_experts) >= num_experts:
        return None  # not an error: nothing to cache, handled by the caller
    return None


def _is_offloaded(layer, *names) -> bool:
    """True if these expert weights actually live in host memory.

    Caching a layer whose weights never left the GPU is worse than pointless: the slot
    buffers cost real VRAM (12.38 MiB per slot per layer on GLM-5.3-Flash) and every step
    still runs a manager and a gather that copy VRAM to VRAM for a lookup that cannot
    miss. Measured on a 42-layer run with a 28 GiB offload budget, 25 of 42 layers were
    fully resident and held 2.45 GiB/rank of slots they could never use.

    vLLM's UVA offloader leaves `.device` reading as the accelerator, because an offloaded
    parameter becomes a device-addressable VIEW of pinned host memory. Its own marker is
    the only reliable signal, so read that, and consult the device only for the non-UVA
    path, which really does move the tensor to CPU.
    """
    for name in names:
        t = getattr(layer, name, None)
        if t is None:
            continue
        if getattr(t, "_vllm_is_uva_offloaded", False) or t.device.type == "cpu":
            return True
    return False


def _weight_names(layer) -> tuple[str, str] | None:
    """The registered names of the two expert weight matrices, whatever they are called.

    Most schemes register `w13_weight`/`w2_weight`, but the packed compressed-tensors
    family (wNa16 int4, w4a4-nvfp4, w4a4-mxfp4) registers `w13_weight_packed`/
    `w2_weight_packed` instead. Resolve by probing rather than assuming, so a new
    spelling is a miss to report rather than a silent skip.
    """
    for a, b in (("w13_weight", "w2_weight"),
                 ("w13_weight_packed", "w2_weight_packed")):
        if isinstance(getattr(layer, a, None), torch.Tensor) and \
           isinstance(getattr(layer, b, None), torch.Tensor):
            return a, b
    return None


def arm(layer, method, logger) -> bool:
    """Attach a slot cache to one MoE layer. Returns True if the cache is live."""
    names = _weight_names(layer)
    if names is None:
        # Say what was actually there. "w13_weight is missing" sends the next person
        # looking for a bug in their checkpoint rather than at a naming convention.
        present = sorted(n for n, _ in layer.named_parameters(recurse=False))
        if any("w13" in n or "w2" in n for n in present):
            logger.warning(
                "expert-cache: not caching %s -- no recognised expert weight pair; the "
                "layer registers %s", type(method).__name__, ", ".join(present) or "nothing")
        return False
    w13_name, w2_name = names

    if (os.environ.get("EXPERT_CACHE_DECLINE_RESIDENT", "1") != "0"
            and not _is_offloaded(layer, w13_name, w2_name)):
        # Not an error, and not silent: which layers the offloader reached depends on
        # --cpu-offload-gb, so this line is how you see that the budget stopped short.
        logger.info(
            "expert-cache: not caching %s -- its experts are still GPU-resident, so a "
            "cache would only spend VRAM and copy weights to themselves. Raise "
            "--cpu-offload-gb to put this layer on the host.", type(method).__name__)
        return False

    num_experts = getattr(layer, w13_name).size(0)
    slots = settings.slots_for(num_experts)
    if slots >= num_experts:
        return False  # everything already fits; the cache would only duplicate VRAM

    reason = _decline(layer, method, num_experts)
    if reason is not None:
        logger.warning("expert-cache: not caching %s -- %s",
                       type(method).__name__, reason)
        return False

    sources = _per_expert_tensors(layer, num_experts)
    reachable = {n: key for key, group in sources.items() for n in group}
    w13_key, w2_key = reachable.get(w13_name), reachable.get(w2_name)
    for name in (w13_name, w2_name):
        if name not in reachable:
            logger.warning(
                "expert-cache: not caching %s -- %s is not a directly-owned expert-major "
                "tensor (found: %s)", type(method).__name__, name,
                ", ".join(sorted(reachable)) or "none")
            return False

    # Slots live wherever the weights are at load time -- which is the accelerator, since
    # the offloader only relocates them to host memory afterwards. Taking the device from
    # the tensor rather than torch.cuda.current_device() keeps this honest if the weights
    # are somewhere else, and is one less CUDA/HIP assumption in the Python layer.
    try:
        cache = ExpertSlotCache(layer, sources.keys(), num_experts, slots,
                                getattr(layer, w13_name).device,
                                required=set(sources[w13_key]) | set(sources[w2_key]))
    except ValueError as e:
        # A layout this cache cannot move correctly (currently: a per-expert slab that is
        # not 16-byte aligned). Declining is a slowdown; guessing would be wrong answers.
        logger.warning("expert-cache: not caching %s -- %s", type(method).__name__, e)
        return False

    # Pre-wrap as Parameters once. Doing it per call would churn objects on the hot path,
    # and vLLM notes elsewhere that graph capture pins parameter storage addresses.
    # Every alias of a storage gets the *same* slot buffer, wrapped as a Parameter only
    # for the names actually registered as parameters -- assigning a bare Tensor over a
    # registered Parameter raises, and wrapping a plain attribute would be a lie.
    cache.bound = {}
    for canonical, group in sources.items():
        if canonical not in cache.slots:
            continue  # skipped above as not 16-byte aligned; left on full storage
        slot = cache.slots[canonical]
        as_param = torch.nn.Parameter(slot, requires_grad=False)
        for name in group:
            cache.bound[name] = as_param if name in layer._parameters else slot

    if cache.skipped:
        logger.info("expert-cache: not mirroring %s (not 16-byte aligned per expert); the "
                    "leak check below decides whether that matters",
                    ", ".join(cache.skipped))

    with _slot_bound(layer, cache):
        slot_cfg = method.get_fused_moe_quant_config(layer)

    # Wide (prefill) steps. The scratch is full-size and indexed by local expert id, so
    # unlike the slot path NOTHING here is remapped -- which is also why the leak check
    # below must not run against it: every per-expert tensor staying at [E] is correct.
    scratch = wide_cfg = None
    if settings.wide_scratch:
        try:
            scratch = WideScratch.acquire(cache, cache.table.device)
            if not scratch.bound:
                for first, group in sources.items():
                    if first not in scratch.buf:
                        continue
                    buf = scratch.buf[first]
                    as_param = torch.nn.Parameter(buf, requires_grad=False)
                    for name in group:
                        scratch.bound[name] = (
                            as_param if name in layer._parameters else buf)
            with _bound_to(layer, scratch.bound):
                wide_cfg = method.get_fused_moe_quant_config(layer)
        except Exception as e:
            # Disclosed, not silent: without this the layer still computes correctly,
            # it just reads the host copies through the GEMM the slow way.
            logger.warning(
                "expert-cache: wide-step scratch unavailable (%r); prefill will read "
                "through to host memory. Set EXPERT_CACHE_WIDE_SCRATCH=0 to silence.", e)
            scratch = wide_cfg = None

    if slot_cfg is not None:
        leaked = _leaked_full_tensors(slot_cfg, num_experts)
        if leaked:
            logger.warning(
                "expert-cache: not caching %s -- these per-expert tensors stayed bound to "
                "full [%d] storage after slot binding and would be indexed by slot id: %s",
                type(method).__name__, num_experts, ", ".join(leaked))
            return False

    orig_apply = method.apply
    experts_ref = method.moe_kernel.fused_experts

    # Parameter names must match the base class exactly: vLLM calls this by keyword
    # (apply(layer=..., x=..., ...)), so renaming even the first one breaks the call.
    def apply(layer, x, topk_weights, topk_ids, shared_experts=None,
              shared_experts_input=None):
        if not cache.fits(topk_ids):
            # Wide / prefill steps touch more distinct experts than there are slots, so
            # they cannot be served from the slot table. Stage the whole local expert set
            # into VRAM with the gather kernel and compute from there: same bytes over
            # the same link, but ~24 GB/s of wide contiguous copies instead of the GEMM
            # picking at host memory at ~7 GB/s. Falls back to the stock read-through
            # path when no scratch was allocated.
            if scratch is None:
                return orig_apply(layer, x, topk_weights, topk_ids,
                                  shared_experts, shared_experts_input)
            scratch.fill(layer, cache.chunks, cache.lanes)
            w_saved = (experts_ref.quant_config, method.moe_quant_config)
            if wide_cfg is not None:
                experts_ref.quant_config = wide_cfg
                method.moe_quant_config = wide_cfg
            try:
                with _bound_to(layer, scratch.bound):
                    return orig_apply(layer, x, topk_weights, topk_ids,
                                      shared_experts, shared_experts_input)
            finally:
                experts_ref.quant_config, method.moe_quant_config = w_saved

        emap = getattr(layer, "expert_map", None)
        experts = method.moe_kernel.fused_experts
        saved_cfg = (experts.quant_config, method.moe_quant_config)
        saved_n = layer.global_num_experts
        saved_emap = getattr(layer, "_expert_map", None)

        if emap is None:
            # No expert parallelism: routing ids are already local, so rewrite them
            # into slot space directly and tell the layer how many experts that is.
            call_ids = cache.refresh(topk_ids)
            composed = None
            n_owned = topk_ids
        else:
            # Expert parallelism. expert_map sends global -> local, or -1 for experts
            # this rank does not own. Rather than remap the ids (which would mean
            # reimplementing the -1 handling), fold the slot table INTO expert_map so
            # the kernel applies one composed global -> slot map and keeps its own
            # not-owned semantics. Routing ids stay global, so global_num_experts must
            # keep describing the global id space.
            #
            # The manager is fed the local ids with their -1s INTACT. Masking them out
            # would make the output shape depend on device data and stall the pipeline
            # once per MoE layer per token. Clamping them to 0 -- what this used to do --
            # avoided that stall but handed the manager one phantom request for local
            # expert 0 per non-owned slot, on every call. At top_k=8 with half the
            # experts non-owned that is ~4 phantoms against ~4 real ids: expert 0 became
            # permanently the hottest entry, pinning a slot, and the LFU counts it sorts
            # on were mostly noise (measured 2026-09-15: hit rate 35%, and lfu beat lru
            # by 0.5pp because neither policy had usable statistics). Passing the -1s
            # through costs nothing: expert_cache_manage_k documents its ids as
            # `<0 = padding` and already guards every read with `e >= 0 && e < E`.
            # expert_map is fixed for the life of the layer, so its derived index
            # and not-owned mask are built once here, not rebuilt per token. Doing
            # it per call cost several elementwise kernels and allocations on every
            # one of the model's MoE layers, every step.
            cached = getattr(layer, "_ec_emap_cache", None)
            if cached is None or cached[0] is not emap:
                out = torch.empty_like(emap)
                layer._ec_emap_cache = (emap, out, {})
                cached = layer._ec_emap_cache
            _, composed, local_bufs = cached

            if _FUSED_REMAP:
                local = fused_remap.gather_local(emap, topk_ids, local_bufs)
            else:
                local = emap[topk_ids.to(torch.int64)]
            if _timing_every:
                _t0 = torch.cuda.Event(enable_timing=True); _t0.record()
            cache.refresh(local)
            if _timing_every:
                _t_ref = torch.cuda.Event(enable_timing=True); _t_ref.record()
            # One kernel into a persistent buffer. The gather and the not-owned mask
            # are the same pass; at 288 experts these were dispatch, not work.
            if _FUSED_REMAP:
                fused_remap.compose(cache.table, emap, composed)
            else:
                torch.index_select(cache.table, 0,
                                   emap.clamp(min=0).to(torch.int64), out=composed)
                composed.masked_fill_(emap < 0, -1)
            call_ids = topk_ids
            n_owned = local

        if _stats_every:
            _report_stats(cache, n_owned, logger)

        experts.quant_config = slot_cfg
        method.moe_quant_config = slot_cfg
        if composed is None:
            layer.global_num_experts = cache.S
        else:
            layer._expert_map = composed
        try:
            with _slot_bound(layer, cache):
                if _timing_every:
                    _t1 = torch.cuda.Event(enable_timing=True); _t1.record()
                    if composed is not None:
                        _TIMING["pend_remap"].append((_t0, _t_ref, _t1))
                    out = orig_apply(layer, x, topk_weights, call_ids,
                                     shared_experts, shared_experts_input)
                    _t2 = torch.cuda.Event(enable_timing=True); _t2.record()
                    _TIMING["pend"].append((_t0, _t1, _t2))
                    _TIMING["n"] += 1
                    if _TIMING["n"] % _timing_every == 0:
                        _drain_timing(logger)
                    return out
                return orig_apply(layer, x, topk_weights, call_ids,
                                  shared_experts, shared_experts_input)
        finally:
            experts.quant_config, method.moe_quant_config = saved_cfg
            layer.global_num_experts = saved_n
            if composed is not None:
                layer._expert_map = saved_emap

    method.apply = apply
    layer._expert_cache = cache
    return True


def _install_offload_order(logger) -> None:
    """Choose WHICH layers the offloader sends to host, instead of taking construction order.

    vLLM offloads modules in the order they are built until a byte budget runs out, so
    which layers end up on the host is an accident. That would be fine if layers were
    interchangeable, but they are not: measured on a real routing trace at 46 slots,
    per-layer LFU hit rate ranges from 46% to 89%, shallow layers routing almost
    uniformly while deep ones concentrate on a stable favourite set. A layer that caches
    well is CHEAP to offload; a flat-routing one is expensive. The default order offloads
    the flat ones first.

    EXPERT_CACHE_OFFLOAD_SKIP=<n>: do not offload the first n modules, so the budget is
    spent on the deeper ones instead. 0 (default) leaves vLLM's behaviour untouched.

    DO NOT reorder by materialising the generator. `modules_generator` BUILDS each layer
    as it is pulled, and the stock list comprehension frees a layer's VRAM by offloading
    it before the next is created. Calling list() on it first allocates every layer on
    the GPU at once -- 63.23 GiB, then OOM partway through creating expert weights.
    Deciding per module as it arrives keeps that lazy property intact.
    """
    try:
        skip = int(os.environ.get("EXPERT_CACHE_OFFLOAD_SKIP", "0"))
    except ValueError:
        skip = 0
    if skip <= 0:
        return
    try:
        from vllm.model_executor.offloader.uva import UVAOffloader
        from vllm.utils.mem_utils import format_gib
    except Exception as e:
        logger.warning("expert-cache: cannot steer offload placement (%r)", e)
        return

    state = {"n": 0}

    def wrap_modules(self, modules_generator, prefix: str = ""):
        pfx = f"{prefix}." if prefix else ""
        out = []
        for module in modules_generator:          # stays lazy -- see the docstring
            i = state["n"]
            state["n"] += 1
            if i < skip:
                out.append(module)                # held resident on purpose
            else:
                out.append(self._maybe_offload_to_cpu(module, pfx))
        if self.cpu_offload_bytes > 0:
            logger.info("expert-cache: held the first %d modules resident; %s offloaded",
                        skip, format_gib(self.cpu_offload_bytes))
        return out

    UVAOffloader.wrap_modules = wrap_modules
    logger.info("expert-cache: offload placement will skip the first %d modules", skip)


def install(logger) -> bool:
    """Patch vLLM's single post-load hook so every MoE layer is armed after loading."""
    global _installed
    if _installed:
        return True
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )
        from vllm.model_executor.model_loader import utils as loader_utils
    except Exception as e:
        logger.debug("expert-cache: vLLM MoE interfaces unavailable (%r)", e)
        return False

    _install_offload_order(logger)

    orig = loader_utils.process_weights_after_loading

    def process_weights_after_loading(model, model_config, target_device):
        orig(model, model_config, target_device)
        if settings.disabled:
            return
        armed = declined = 0
        first = None
        methods: set[int] = set()
        for _, module in model.named_modules():
            method = getattr(module, "quant_method", None)
            if not isinstance(method, FusedMoEMethodBase):
                continue
            if id(method) in methods:
                # vLLM builds one quant_method per layer. If that ever stops being true,
                # wrapping the same instance twice would bind its slot config to the
                # first layer's buffers and quietly serve one layer's experts from
                # another's, so skip loudly instead.
                logger.warning(
                    "expert-cache: %s instance is shared across layers; leaving the "
                    "extra layers uncached", type(method).__name__)
                declined += 1
                continue
            try:
                if arm(module, method, logger):
                    armed += 1
                    methods.add(id(method))
                    first = first or module._expert_cache
                else:
                    declined += 1
            except Exception as e:
                # Never break serving: without a cache the layer reads through as before.
                logger.warning("expert-cache: disabled for layer (%r)", e)
                module._expert_cache = None
                declined += 1
        if first is not None:
            logger.info(
                "expert-cache %s active on %d MoE layer(s) (%d declined): %d/%d experts "
                "resident, mirroring %s, policy=%s",
                _version(), armed, declined, first.S, first.E,
                "+".join(first.names), settings.policy)
        else:
            logger.info("expert-cache: no cacheable MoE layer found; staying out of the way")

    # Rebind at the source *and* in every module that already did
    # `from ...utils import process_weights_after_loading` -- base_loader does exactly
    # that, so patching only `utils` leaves the real call site pointing at the original
    # and the plugin silently does nothing. Patching `utils` as well covers importers
    # that have not been loaded yet.
    patched = []
    for name, mod in list(sys.modules.items()):
        if mod is not None and getattr(mod, "process_weights_after_loading", None) is orig:
            setattr(mod, "process_weights_after_loading", process_weights_after_loading)
            patched.append(name.rsplit(".", 1)[-1])
    loader_utils.process_weights_after_loading = process_weights_after_loading

    logger.info("expert-cache: hooked the MoE loader in %s", ", ".join(patched) or "utils")
    _installed = True
    return True


def _version() -> str:
    from .. import __version__
    return __version__
