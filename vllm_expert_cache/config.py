"""Runtime configuration, all via environment variables."""

from __future__ import annotations

import os


class _Settings:
    @property
    def disabled(self) -> bool:
        return os.environ.get("EXPERT_CACHE_DISABLE", "0") == "1"

    @property
    def slots(self) -> int | None:
        """Absolute slot count, if set."""
        v = os.environ.get("EXPERT_CACHE_SLOTS")
        return int(v) if v else None

    @property
    def fraction(self) -> float:
        """Fraction of experts to keep resident when EXPERT_CACHE_SLOTS is unset."""
        return float(os.environ.get("EXPERT_CACHE_FRACTION", "0.5"))

    @property
    def policy(self) -> str:
        """Replacement policy: 'lfu' (frequency with decay, default) or 'lru' (recency).

        LFU fetches 11-26% fewer experts across budgets and workloads; both rules are
        validated bit-exactly against the reference model in tests/test_policy.py.
        """
        return os.environ.get("EXPERT_CACHE_POLICY", "lfu").strip().lower()

    @property
    def policy_code(self) -> int:
        return {"lru": 0, "lfu": 1}.get(self.policy, 0)

    @property
    def wide_scratch(self) -> bool:
        """Stage all local experts into VRAM for steps too wide to serve from slots.

        Costs one full-size buffer per layer geometry (E x per-expert bytes, shared by
        every layer of that shape), and buys the difference between the GEMM reading
        host memory at ~7 GB/s and the gather kernel streaming it at ~24 GB/s.
        """
        return os.environ.get("EXPERT_CACHE_WIDE_SCRATCH", "1") != "0"

    @property
    def decay(self) -> int:
        """LFU only: halve every N steps so stale-hot experts age out. 0 disables."""
        return int(os.environ.get("EXPERT_CACHE_DECAY", "64"))

    def slots_for(self, num_experts: int) -> int:
        n = self.slots if self.slots is not None else round(num_experts * self.fraction)
        return max(1, min(int(n), num_experts))


settings = _Settings()
