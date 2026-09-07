"""Runtime configuration, all via environment variables."""

from __future__ import annotations

import os


class _Settings:
    @property
    def disabled(self) -> bool:
        return os.environ.get("LRU_CACHE_DISABLE", "0") == "1"

    @property
    def slots(self) -> int | None:
        """Absolute slot count, if set."""
        v = os.environ.get("LRU_CACHE_SLOTS")
        return int(v) if v else None

    @property
    def fraction(self) -> float:
        """Fraction of experts to keep resident when LRU_CACHE_SLOTS is unset."""
        return float(os.environ.get("LRU_CACHE_FRACTION", "0.5"))

    @property
    def policy(self) -> str:
        """Replacement policy: 'lru' (recency) or 'lfu' (frequency with decay)."""
        return os.environ.get("LRU_CACHE_POLICY", "lru").strip().lower()

    @property
    def policy_code(self) -> int:
        return {"lru": 0, "lfu": 1}.get(self.policy, 0)

    @property
    def decay(self) -> int:
        """LFU only: halve every N steps so stale-hot experts age out. 0 disables."""
        return int(os.environ.get("LRU_CACHE_DECAY", "64"))

    def slots_for(self, num_experts: int) -> int:
        n = self.slots if self.slots is not None else round(num_experts * self.fraction)
        return max(1, min(int(n), num_experts))


settings = _Settings()
