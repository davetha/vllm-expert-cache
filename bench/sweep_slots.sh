#!/usr/bin/env bash
# Sweep the resident-expert budget and report decode throughput for each.
#
#   MODEL=/models/qwen3-30b-a3b-w8a8 ./sweep_slots.sh 96 64 48 32
#
# Serves the model once per budget with the cache armed, warms it, then times a fixed
# 200-token greedy completion. Compare against VLLM_LRU_DISABLE=1 for the read-through
# floor and a no-offload run for the fully-resident ceiling.
set -uo pipefail
MODEL=${MODEL:?set MODEL=/path/to/checkpoint}
PORT=${PORT:-8075}
OFFLOAD=${OFFLOAD:-22}
PROMPT=${PROMPT:-"Write a detailed paragraph about the history of computing."}
BUDGETS=("$@"); [ ${#BUDGETS[@]} -eq 0 ] && BUDGETS=(96 64 48 32)

for S in "${BUDGETS[@]}"; do
  VLLM_LRU_SLOTS=$S vllm serve "$MODEL" --port "$PORT" \
    --tensor-parallel-size "${TP:-2}" --max-model-len "${LEN:-8192}" \
    --cpu-offload-gb "$OFFLOAD" --cpu-offload-params experts >/tmp/lru-$S.log 2>&1 &
  pid=$!
  for _ in $(seq 1 60); do sleep 10; grep -qa "Application startup complete" /tmp/lru-$S.log && break; done
  curl -s "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"warm\",\"max_tokens\":40}" >/dev/null
  curl -s "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":200,\"temperature\":0}" \
    -w '\nT=%{time_total}\n' | awk -v s="$S" '
      /"completion_tokens"/{match($0,/"completion_tokens":[ ]*([0-9]+)/,m); ct=m[1]}
      /^T=/{split($0,a,"="); printf "slots=%s  %.1f tok/s\n", s, ct/a[2]}'
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
done
