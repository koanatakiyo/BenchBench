#!/usr/bin/env bash
# Re-run Stage 3 dynamic quality pass for specific designer/answerer pairs.
# Uses --skip-existing to avoid re-answering; only dynamic/judge recomputes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

CONFIG="${CONFIG:-${ROOT_DIR}/benchbench/config/stage3.yaml}"
DATASET="${DATASET:-csbench_en}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

declare -a DESIGNERS=(
  "claude_sonnet_45"
  "deepseek_chat"
  "doubao_seed_1_6_flash"
  "gemini_2_5_flash"
  "qwen3_next_80b_a3b_instruct"
)

declare -a ANSWERERS=(
  "llama4_maverick"
)

# Meta-llama floor variants vs other designers
declare -a EXTRA_DESIGNERS=(
  "claude_sonnet_45"
  "deepseek_v3_2_chat"
  "doubao_seed_1_6_flash"
  "qwen3_next_80b_a3b_instruct"
)
EXTRA_ANSWERER="meta_llama_llama_4_maverick_floor"

cd "${SCRIPT_DIR}"

echo "Using config: ${CONFIG}"
echo "Dataset: ${DATASET}"
echo "Log level: ${LOG_LEVEL}"

run_pair() {
  local designer="$1"
  local answerer="$2"
  echo "== Re-running dynamic pass for designer=${designer} answerer=${answerer} =="
  python run_stage3.py \
    --config "${CONFIG}" \
    --dataset "${DATASET}" \
    --designer "${designer}" \
    --answerer "${answerer}" \
    --skip-existing \
    --log-level "${LOG_LEVEL}"
}

for d in "${DESIGNERS[@]}"; do
  for a in "${ANSWERERS[@]}"; do
    run_pair "${d}" "${a}"
  done
done

for d in "${EXTRA_DESIGNERS[@]}"; do
  run_pair "${d}" "${EXTRA_ANSWERER}"
done

echo "Done."

