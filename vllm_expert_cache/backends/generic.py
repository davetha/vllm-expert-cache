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
import sys

import torch

from ..cache import ExpertSlotCache
from ..config import settings

# Fields on a FusedMoEQuantDesc that can hold a per-expert tensor.
_DESC_FIELDS = ("scale", "alpha_or_gscale", "zp", "bias")
_DESC_NAMES = ("_a1", "_a2", "_w1", "_w2")

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
def _slot_bound(layer, cache: ExpertSlotCache):
    """Point every mirrored attribute at its slot buffer for the duration of the block."""
    saved = {name: getattr(layer, name) for name in cache.bound}
    try:
        for name, slot in cache.bound.items():
            _set_tensor(layer, name, slot)
        yield
    finally:
        for name, t in saved.items():
            _set_tensor(layer, name, t)


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
    if getattr(layer, "expert_map", None) is not None:
        # With expert parallelism topk_ids are global ids that expert_map folds into a
        # local range; composing that with the slot table is a second remap this does
        # not implement.
        return "expert parallelism active (layer.expert_map is set)"
    if settings.slots_for(num_experts) >= num_experts:
        return None  # not an error: nothing to cache, handled by the caller
    return None


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

    if slot_cfg is not None:
        leaked = _leaked_full_tensors(slot_cfg, num_experts)
        if leaked:
            logger.warning(
                "expert-cache: not caching %s -- these per-expert tensors stayed bound to "
                "full [%d] storage after slot binding and would be indexed by slot id: %s",
                type(method).__name__, num_experts, ", ".join(leaked))
            return False

    orig_apply = method.apply

    # Parameter names must match the base class exactly: vLLM calls this by keyword
    # (apply(layer=..., x=..., ...)), so renaming even the first one breaks the call.
    def apply(layer, x, topk_weights, topk_ids, shared_experts=None,
              shared_experts_input=None):
        if not cache.fits(topk_ids):
            # Wide / prefill steps touch more distinct experts than there are slots and
            # read through the host copies on the stock path.
            return orig_apply(layer, x, topk_weights, topk_ids,
                              shared_experts, shared_experts_input)

        slot_ids = cache.refresh(topk_ids)
        experts = method.moe_kernel.fused_experts
        saved = (experts.quant_config, method.moe_quant_config,
                 layer.global_num_experts)
        experts.quant_config = slot_cfg
        method.moe_quant_config = slot_cfg
        layer.global_num_experts = cache.S
        try:
            with _slot_bound(layer, cache):
                return orig_apply(layer, x, topk_weights, slot_ids,
                                  shared_experts, shared_experts_input)
        finally:
            (experts.quant_config, method.moe_quant_config,
             layer.global_num_experts) = saved

    method.apply = apply
    layer._expert_cache = cache
    return True


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
