#!/usr/bin/env bash
# Simple helper to launch a vLLM OpenAI-compatible server for Stage 2 runs.
#
# Usage:
#   ./run_vllm_server.sh [hf_model_id]
# Environment overrides:
#   PORT (default 8000)               – HTTP port for the OpenAI-compatible API
#   HOST (default 0.0.0.0)            – Bind address
#   TP_SIZE (default 2)               – Tensor parallelism degree
#   MAX_MODEL_LEN (default 4096)      – Maximum context length
#   QUANTIZATION (default awq)        – vLLM quantization flag (awq/gptq/none)
#   HF_DOWNLOAD_DIR                   – Directory to store downloaded weights
#
# Example (AWQ 4-bit Llama 3.3 70B):
#   PORT=9000 TP_SIZE=4 ./run_vllm_server.sh casperhansen/llama-3.3-70b-instruct-awq
#
# Once the server is up, point Stage 2 at it via:
#   python run_stage2.py ... --model vllm --llm-model-name casperhansen/llama-3.3-70b-instruct-awq
set -euo pipefail

MODEL_ID="${1:-casperhansen/llama-3.3-70b-instruct-awq}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
QUANTIZATION="${QUANTIZATION:-awq}"
HF_DOWNLOAD_DIR="${HF_DOWNLOAD_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}"

echo "[vLLM] Starting server with model: ${MODEL_ID}"
echo "       host=${HOST} port=${PORT} tp=${TP_SIZE} max_len=${MAX_MODEL_LEN} quant=${QUANTIZATION}"
echo "       download_dir=${HF_DOWNLOAD_DIR}"

CMD=(python -m vllm.entrypoints.openai.api_server
     --model "${MODEL_ID}"
     --quantization "${QUANTIZATION}"
     --host "${HOST}"
     --port "${PORT}"
     --tensor-parallel-size "${TP_SIZE}"
     --max-model-len "${MAX_MODEL_LEN}")

if [[ -n "${HF_DOWNLOAD_DIR}" ]]; then
    CMD+=(--download-dir "${HF_DOWNLOAD_DIR}")
fi

exec "${CMD[@]}"

