# Measured results

## Setup

- **Hardware**: 2x AMD Instinct MI210 (gfx90a / CDNA2), PCIe Gen4, no XGMI bridge
- **Model**: Qwen3-30B-A3B-Instruct-2507, compressed-tensors W8A8 int8 (128 experts, top-8, 48 layers)
- **vLLM**: int8 fused-MoE path (`TritonInt8ScaledMMLinearKernel` + Triton int8 MoE), TP2,
  `--max-model-len 8192`, `--gpu-memory-utilization 0.85`, HIP graphs enabled
- **Offload**: `--cpu-offload-gb 22 --cpu-offload-params experts` (~13.6 GB of experts per worker
  moved to host memory)
- **Measurement**: single stream, greedy, 200-token completion, wall-clock; warmed with a
  40-token request first. Output checked for coherence at every configuration.

## Decode throughput vs resident-expert budget

| Experts in VRAM | Decode (tok/s) | vs resident |
| --- | --- | --- |
| 128 (no offload) | 65.7 | 100% |
| 96 | 58.4 | 89% |
| 64 | 54.3 | 83% |
| 48 | 49.3 | 75% |
| 32 | 39.4 | 60% |
| 0 (offload, cache disabled) | 19.1 | 29% |

Returns diminish as the budget grows: the first 32 slots roughly double throughput over the
un-cached floor, while the last 32 (96 -> 128) buy about 7 tok/s.

## Static hot set, for comparison

A static set of 64 experts (warm, then freeze, read through on a miss) measures **13.9 tok/s**
against the LRU's **13.7 tok/s** in eager mode — a tie, because eager per-layer dispatch cost
dominates and hides cache behaviour entirely.

The difference only appears with graphs enabled, and there the static design cannot compete:
its per-step residency check is a device-to-host sync that breaks graph capture, so it stays
near the eager number while the LRU captures and reaches 54.3.

## Kernel cost

From `tests/test_policy.py` on one MI210:

- `lru_manage`: ~11.3 us per layer-step at steady state
- `lru_gather`: 27.1 GB/s sustained — the PCIe Gen4 host-to-device roofline on this box
- Policy correctness: matches an independent numpy LRU reference on every field, every step,
  across five routing patterns; gathered bytes are identical to their source rows.
