"""LRU expert cache for compressed-tensors W8A8 int8 MoE layers.

Wraps `CompressedTensorsW8A8Int8MoEMethod`. The int8 fused-MoE kernel indexes weights and
their per-channel scales with the same expert id, so both have to be slot-indexed: the
cache mirrors four per-expert tensors (w13/w2 and their scales) into VRAM slots.

Scales reach the kernel through a `FusedMoEQuantConfig` whose `w1_scale`/`w2_scale` are
read-only properties, so they cannot be swapped per call. Instead a second kernel is built
once at load time, bound to the slot scale buffers, and used for every cached step.
"""

from __future__ import annotations

import torch

from ..cache import ExpertSlotCache
from ..config import settings

# The four per-expert tensors the int8 path needs, in gather order.
SOURCES = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")

_installed = False


def install(logger) -> bool:
    global _installed
    if _installed:
        return True
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (  # noqa: E501
            CompressedTensorsW8A8Int8MoEMethod,
            make_int8_moe_kernel,
        )
    except Exception as e:
        logger.debug("lru-expert-cache: int8 MoE backend unavailable (%r)", e)
        return False

    orig_pwal = CompressedTensorsW8A8Int8MoEMethod.process_weights_after_loading
    orig_apply = CompressedTensorsW8A8Int8MoEMethod.apply

    def _build_slot_kernel(self, layer, cache):
        """A kernel bound to the slot scale buffers rather than the full [E] scales."""
        keep13 = layer.w13_weight_scale
        keep2 = layer.w2_weight_scale
        layer.w13_weight_scale = torch.nn.Parameter(
            cache.slots["w13_weight_scale"], requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(
            cache.slots["w2_weight_scale"], requires_grad=False)
        try:
            return make_int8_moe_kernel(
                int8_backend=self.int8_backend,
                moe_quant_config=self.get_fused_moe_quant_config(layer),
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
        finally:
            layer.w13_weight_scale = keep13
            layer.w2_weight_scale = keep2

    def process_weights_after_loading(self, layer):
        orig_pwal(self, layer)
        if settings.disabled:
            return
        try:
            num_experts = layer.w13_weight.size(0)
            slots = settings.slots_for(num_experts)
            if slots >= num_experts:
                return  # nothing to cache: everything is already resident
            device = torch.device("cuda", torch.cuda.current_device())
            cache = ExpertSlotCache(layer, SOURCES, num_experts, slots, device)
            layer._lru_slot_kernel = _build_slot_kernel(self, layer, cache)
            layer._lru_cache = cache
            logger.info("lru-expert-cache: armed layer with %d/%d experts resident",
                        slots, num_experts)
        except Exception as e:
            # Never break serving: without a cache the layer just reads through as before.
            logger.warning("lru-expert-cache: disabled for layer (%r)", e)
            layer._lru_cache = None

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        cache = getattr(layer, "_lru_cache", None)
        if cache is None or not cache.fits(topk_ids):
            # Wide / prefill steps touch more distinct experts than there are slots;
            # those read through the host copies on the stock path.
            return orig_apply(self, layer, x, topk_weights, topk_ids,
                              shared_experts, shared_experts_input)
        slot_ids = cache.refresh(topk_ids)
        return layer._lru_slot_kernel.apply(
            x,
            cache.slots["w13_weight"],
            cache.slots["w2_weight"],
            topk_weights=topk_weights,
            topk_ids=slot_ids,
            activation=layer.activation,
            global_num_experts=cache.S,
            expert_map=None,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    CompressedTensorsW8A8Int8MoEMethod.process_weights_after_loading = process_weights_after_loading
    CompressedTensorsW8A8Int8MoEMethod.apply = apply
    _installed = True
    return True
