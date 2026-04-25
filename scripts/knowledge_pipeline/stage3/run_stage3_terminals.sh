#!/usr/bin/env bash
# Generate commands to run stage3 answerers in separate terminal windows
# Usage:
#   ./run_stage3_terminals.sh [dataset] [--skip-existing] [--dry-run]
# 
# This will print commands that you can copy-paste into separate terminal windows
# to run each answer model in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE3_SCRIPT="${SCRIPT_DIR}/run_stage3.py"
# Config is at benchbench/config/stage3.yaml relative to workspace root
CONFIG="${SCRIPT_DIR}/../../../config/stage3.yaml"

# Parse arguments
DATASET="${1:-}"
SKIP_EXISTING=""
DRY_RUN=""
OTHER_ARGS=()

# Check if first arg is a flag
if [[ "${1:-}" =~ ^-- ]]; then
    DATASET=""
else
    shift 2>/dev/null || true
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-existing)
            SKIP_EXISTING="--skip-existing"
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Extract answer model keys from config
extract_answer_models() {
    local config_file="$1"
    python3 << PYTHON
import yaml
import sys
from pathlib import Path

config_path = Path("${config_file}")
with open(config_path) as f:
    config = yaml.safe_load(f)

answer_models = config.get('stage3', {}).get('answer_models', {})
for key in answer_models.keys():
    print(key)
PYTHON
}

# Get answer models
if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: Config file not found: ${CONFIG}" >&2
    exit 1
fi

ANSWER_MODELS=($(extract_answer_models "${CONFIG}"))

if [[ ${#ANSWER_MODELS[@]} -eq 0 ]]; then
    echo "Error: No answer models found in config" >&2
    exit 1
fi

echo "=================================================================================="
echo "Stage3 Terminal Commands Generator"
echo "=================================================================================="
echo "Dataset: ${DATASET:-<from config>}"
echo "Answer models: ${#ANSWER_MODELS[@]}"
echo ""
echo "Copy each command below into a separate terminal window:"
echo "=================================================================================="
echo ""

# Build base command
BASE_CMD="cd $(pwd) && python3 ${STAGE3_SCRIPT} --config ${CONFIG}"
if [[ -n "${DATASET}" ]]; then
    BASE_CMD="${BASE_CMD} --dataset ${DATASET}"
fi
BASE_CMD="${BASE_CMD} --skip-dynamic"  # Skip dynamic during individual runs
if [[ -n "${SKIP_EXISTING}" ]]; then
    BASE_CMD="${BASE_CMD} ${SKIP_EXISTING}"
fi
if [[ -n "${DRY_RUN}" ]]; then
    BASE_CMD="${BASE_CMD} ${DRY_RUN}"
fi
if [[ ${#OTHER_ARGS[@]} -gt 0 ]]; then
    BASE_CMD="${BASE_CMD} ${OTHER_ARGS[*]}"
fi

# Generate commands for each answerer
for i in "${!ANSWER_MODELS[@]}"; do
    answerer="${ANSWER_MODELS[$i]}"
    echo "# Terminal $((i+1)): ${answerer}"
    echo "${BASE_CMD} --answerer ${answerer}"
    echo ""
done

echo "=================================================================================="
echo "After ALL answerers complete, run dynamic quality pass in ONE terminal:"
echo "=================================================================================="
DYN_CMD="cd $(pwd) && python3 ${STAGE3_SCRIPT} --config ${CONFIG}"
if [[ -n "${DATASET}" ]]; then
    DYN_CMD="${DYN_CMD} --dataset ${DATASET}"
fi
DYN_CMD="${DYN_CMD} --skip-existing"
# Note: NO --skip-dynamic here - we want to RUN dynamic quality
echo "${DYN_CMD}"
echo ""
echo "=================================================================================="
echo "Tips:"
echo "  - Each answerer can run in parallel in separate terminals"
echo "  - Monitor progress with: tail -f stage3_<answerer>.log (if logging to file)"
echo "  - Wait for all to finish before running dynamic quality pass"
echo "=================================================================================="

