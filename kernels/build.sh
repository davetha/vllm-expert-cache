#!/usr/bin/env bash
# Build libexpertcache.so for one or more GPU architectures.
#
#   ./build.sh                 # defaults to gfx90a
#   ./build.sh gfx942          # MI300
#   ./build.sh gfx90a gfx1201  # fat binary for CDNA2 + RDNA4
#
# The kernels use only block-level LDS reductions, so one source serves wave64 and wave32.
# Drop the result next to the Python package (vllm_expert_cache/) or point EXPERT_CACHE_LIB at it.
set -euo pipefail
cd "$(dirname "$0")"

ARCHS=("$@"); [ ${#ARCHS[@]} -eq 0 ] && ARCHS=(gfx90a)
OUT=${OUT:-../vllm_expert_cache/libexpertcache.so}

# hipcc lives in a system ROCm install, or in the ROCm pip wheels some images ship.
HIPCC=${HIPCC:-}
if [ -z "$HIPCC" ]; then
  for c in /opt/rocm/bin/hipcc "$(command -v hipcc || true)" \
           /opt/python/lib/python*/site-packages/_rocm_sdk_devel/bin/hipcc; do
    [ -n "$c" ] && [ -x "$c" ] && { HIPCC=$c; break; }
  done
fi
[ -n "$HIPCC" ] || { echo "hipcc not found; set HIPCC=/path/to/hipcc" >&2; exit 1; }

FLAGS=(-O3 -std=c++17 -fPIC -shared)
for a in "${ARCHS[@]}"; do FLAGS+=(--offload-arch="$a"); done

# Wheel-based ROCm needs an explicit device-bitcode path; a system install finds its own.
DEVLIB=$(ls -d /opt/python/lib/python*/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode 2>/dev/null | head -1 || true)
[ -n "$DEVLIB" ] && FLAGS+=(--rocm-device-lib-path="$DEVLIB")

echo "building for: ${ARCHS[*]}"
"$HIPCC" "${FLAGS[@]}" expert_cache.hip -o "$OUT"

for sym in expert_cache_manage expert_cache_gather expert_cache_fused; do
  nm -D "$OUT" | grep -q " T $sym$" || { echo "MISSING EXPORT: $sym" >&2; exit 1; }
done
echo "built $OUT"; ls -l "$OUT"
