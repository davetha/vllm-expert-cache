"""Runtime configuration, all via environment variables."""

from __future__ import annotations

import os


class _Settings:
    @property
    def disabled(self) -> bool:
        return os.environ.get("VLLM_LRU_DISABLE", "0") == "1"

    @property
    def slots(self) -> int | None:
        """Absolute slot count, if set."""
        v = os.environ.get("VLLM_LRU_SLOTS")
        return int(v) if v else None

    @property
    def fraction(self) -> float:
        """Fraction of experts to keep resident when VLLM_LRU_SLOTS is unset."""
        return float(os.environ.get("VLLM_LRU_FRACTION", "0.5"))

    def slots_for(self, num_experts: int) -> int:
        n = self.slots if self.slots is not None else round(num_experts * self.fraction)
        return max(1, min(int(n), num_experts))


settings = _Settings()
