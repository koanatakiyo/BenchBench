#!/usr/bin/env bash
# Helper script to generate Stage 2 questions for multiple providers/models.
# Each model runs sequentially so logs stay readable. Adjust dataset list or
# per-model arguments below as needed.
set -euo pipefail

DATASET="${1:-csbench_en}"
QPC="${QUESTIONS_PER_CALL:-20}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAGE2="${SCRIPT_DIR}/run_stage2.py"

run_model() {
  local provider="$1"
  local model="$2"
  local model_arg="$3"
  echo "=== Running ${provider} :: ${model} ==="
  python "${RUN_STAGE2}" "${DATASET}" \
    --model "${provider}" \
    --llm-model-name "${model_arg}" \
    --questions-per-call "${QPC}" \
    "${@:4}"
}

# 1. Anthropic Claude
# run_model "anthropic" "claude-sonnet-4-5-20250929" "claude-sonnet-4-5-20250929"

# 2. Gemini
run_model "gemini" "gemini-2.5-flash" "gemini-2.5-flash"

# 3. OpenAI GPT-5 mini
# run_model "openai" "gpt-5-mini" "gpt-5-mini"

# # 4. DeepSeek
run_model "deepseek" "deepseek-chat" "deepseek-chat"

# # 5. Meta Llama vLLM API
run_model "llama" "meta-llama/llama-4-maverick:floor" "meta-llama/llama-4-maverick:floor"

# 6. Grok
run_model "grok" "grok-4.1-fast" "grok-4.1-fast"

# 7. Doubao
run_model "doubao" "doubao-seed-1-6-flash-250828" "doubao-seed-1-6-flash-250828"

# 8. Qwen
run_model "qwen" "qwen3-next-80b-a3b-instruct" "qwen3-next-80b-a3b-instruct"

echo "All runs completed."

