"""Device-side LRU expert cache for vLLM MoE models.

Keeps a bounded set of experts resident in VRAM and streams the rest from host memory,
so a mixture-of-experts model that does not fit on the GPU still decodes at close to
fully-resident speed.

vLLM loads this automatically through the `vllm.general_plugins` entry point; installing
the package is enough. Set `EXPERT_CACHE_DISABLE=1` to turn it off without uninstalling.
"""

from __future__ import annotations

from .config import settings

__all__ = ["install", "settings"]
__version__ = "0.1.0"


def install() -> None:
    """Patch every supported MoE quantisation backend. Called by vLLM at startup."""
    try:
        from vllm.logger import init_logger
        logger = init_logger("vllm_expert_cache")
    except Exception:  # pragma: no cover - vLLM always provides this in practice
        import logging
        logger = logging.getLogger("vllm_expert_cache")

    if settings.disabled:
        logger.info("expert-cache: disabled by EXPERT_CACHE_DISABLE")
        return

    from .backends import compressed_tensors_int8

    armed = [name for name, mod in (("compressed-tensors int8", compressed_tensors_int8),)
             if mod.install(logger)]
    if armed:
        logger.info("expert-cache %s active for: %s", __version__, ", ".join(armed))
    else:
        logger.info("expert-cache: no supported MoE backend found; staying out of the way")
