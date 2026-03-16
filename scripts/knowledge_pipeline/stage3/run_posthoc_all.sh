#!/usr/bin/env bash
# Run posthoc_clean_topup for a fixed set of provider dirs/designers.
# Usage:
#   ./run_posthoc_all.sh csbench_en /abs/path/to/outputs/stage2_questions/csbench_en
# Defaults: dataset=csbench_en, root=../outputs/stage2_questions/csbench_en relative to this script.

set -euo pipefail

DATASET="${1:-csbench_en}"
ROOT_ARG="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTHOC="${SCRIPT_DIR}/posthoc_clean_topup.py"

if [[ -n "${ROOT_ARG}" ]]; then
  ROOT_DIR="${ROOT_ARG}"
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.."/outputs/stage2_questions/"${DATASET}" && pwd)"
fi

run_one() {
  local designer="$1"
  local subdir="$2"
  local provider="$3"
  local model_override="$4"
  echo "=== posthoc: designer=${designer} dir=${subdir} provider=${provider} model=${model_override} ==="
  python "${POSTHOC}" \
    --dataset "${DATASET}" \
    --provider-dir "${ROOT_DIR}/${subdir}" \
    --designer-model "${designer}" \
    --llm-provider "${provider}" \
    --llm-model-override "${model_override}"
}

# List: designer, provider-dir, llm-provider, llm-model-override
# run_one "qwen3-next-80b-a3b-instruct" "qwen3-next-80b-a3b-instruct" "qwen" "qwen3-next-80b-a3b-instruct"
# run_one "meta-llama/llama-4-maverick:floor" "meta-llama_llama-4-maverick_floor" "llama" "meta-llama/llama-4-maverick:floor"
run_one "gpt-5-mini" "gpt-5-mini" "openai" "gpt-5-mini"
# run_one "doubao-seed-1-6-flash-250828" "doubao-seed-1-6-flash-250828" "doubao" "doubao-seed-1-6-flash-250828"
# run_one "gemini-2.5-flash" "gemini-2_5-flash" "gemini" "gemini-2.5-flash"
# run_one "deepseek-chat" "deepseek-chat" "deepseek" "deepseek-chat"

echo "All posthoc runs completed."

