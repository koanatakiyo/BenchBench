#!/usr/bin/env bash
# Run Stage 2 for multiple datasets using a single provider/model.
# Usage:
#   ./run_one_model_multi_datasets.sh <provider> <model_name> <dataset1> [dataset2 ...]
# Example:
#   ./run_one_model_multi_datasets.sh openai gpt-4o csbench_en tombench_en wemath_stage2_textonly
#   ./run_one_model_multi_datasets.sh vllm casperhansen/llama-3.3-70b-instruct-awq tombench_cn
# Env overrides:
#   QUESTIONS_PER_CALL (default 40)
#   TEMPERATURE (default 0.8)
#   EXTRA_STAGE2_ARGS='--batch-mode openai --batch-prompts-per-file 15'
#   VLLM_BASE_URL (default http://localhost:8000/v1)
#   VLLM_MAX_TOKENS (default 16000)

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <provider> <model_name> <dataset1> [dataset2 ...]" >&2
  exit 1
fi

PROVIDER="$1"
MODEL_NAME="$2"
shift 2
DATASETS=("$@")

QPC="${QUESTIONS_PER_CALL:-40}"
TEMP="${TEMPERATURE:-0.8}"
VLLM_URL="${VLLM_BASE_URL:-http://localhost:8000/}"
VLLM_MAX="${VLLM_MAX_TOKENS:-16000}"

# Optional extra args passed to run_stage2 (space-separated)
EXTRA_STAGE2_ARGS_ARRAY=()
if [[ -n "${EXTRA_STAGE2_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_STAGE2_ARGS_ARRAY=(${EXTRA_STAGE2_ARGS})
fi

# Auto-append vLLM options when provider is vllm (can be overridden via EXTRA_STAGE2_ARGS)
if [[ "${PROVIDER}" == "vllm" ]]; then
  EXTRA_STAGE2_ARGS_ARRAY+=(
    --vllm-base-url "${VLLM_URL}"
    --vllm-max-tokens "${VLLM_MAX}"
  )
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAGE2="${SCRIPT_DIR}/run_stage2.py"

for ds in "${DATASETS[@]}"; do
  echo "=== Stage 2 :: ${PROVIDER}/${MODEL_NAME} :: ${ds} ==="
  python "${RUN_STAGE2}" "${ds}" \
    --model "${PROVIDER}" \
    --llm-model-name "${MODEL_NAME}" \
    --designer-model "${MODEL_NAME}" \
    --questions-per-call "${QPC}" \
    --temperature "${TEMP}" \
    "${EXTRA_STAGE2_ARGS_ARRAY[@]}"
done

echo "All datasets completed for ${PROVIDER}/${MODEL_NAME}."

