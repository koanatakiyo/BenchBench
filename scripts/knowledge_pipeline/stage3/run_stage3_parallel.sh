#!/usr/bin/env bash
# Run stage3 in parallel by answer model - each answer model runs in its own process
# Usage:
#   ./run_stage3_parallel.sh [dataset] [--skip-dynamic] [--skip-existing] [--dry-run]
# 
# This script will:
#   1. Extract all answer models from config
#   2. Run each answer model in parallel (or provide commands for separate terminals)
#   3. Wait for all to complete
#   4. Run dynamic quality pass (unless --skip-dynamic)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE3_SCRIPT="${SCRIPT_DIR}/run_stage3.py"
# Config is at benchbench/config/stage3.yaml relative to workspace root
CONFIG="${SCRIPT_DIR}/../../../config/stage3.yaml"

# Parse arguments
DATASET="${1:-}"
SKIP_DYNAMIC=""
SKIP_EXISTING=""
DRY_RUN=""
OTHER_ARGS=()

shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-dynamic)
            SKIP_DYNAMIC="--skip-dynamic"
            shift
            ;;
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
echo "Stage3 Parallel Runner"
echo "=================================================================================="
echo "Dataset: ${DATASET:-<from config>}"
echo "Answer models found: ${#ANSWER_MODELS[@]}"
echo "  ${ANSWER_MODELS[*]}"
echo ""
echo "Options:"
echo "  Skip dynamic: ${SKIP_DYNAMIC:-<no>}"
echo "  Skip existing: ${SKIP_EXISTING:-<no>}"
echo "  Dry run: ${DRY_RUN:-<no>}"
echo "=================================================================================="
echo ""

# Function to run one answer model
run_one_answerer() {
    local answerer="$1"
    local log_file="stage3_${answerer}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting answerer: ${answerer}"
    
    # Always skip dynamic quality during individual answerer runs
    # We'll run it once at the end after all answerers complete
    local cmd_args=(
        "${STAGE3_SCRIPT}"
        --config "${CONFIG}"
    )
    
    if [[ -n "${DATASET}" ]]; then
        cmd_args+=(--dataset "${DATASET}")
    fi
    
    cmd_args+=(--answerer "${answerer}")
    cmd_args+=(--skip-dynamic)  # Always skip during individual runs
    
    if [[ -n "${SKIP_EXISTING}" ]]; then
        cmd_args+=("${SKIP_EXISTING}")
    fi
    
    if [[ -n "${DRY_RUN}" ]]; then
        cmd_args+=("${DRY_RUN}")
    fi
    
    cmd_args+=("${OTHER_ARGS[@]}")
    
    python3 "${cmd_args[@]}" > "${log_file}" 2>&1
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed answerer: ${answerer}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed answerer: ${answerer} (exit code: $exit_code)"
        echo "  Check log: ${log_file}"
    fi
    return $exit_code
}

# Check if user wants to run in separate terminals
if [[ "${PARALLEL_MODE:-}" == "terminals" ]]; then
    echo "=================================================================================="
    echo "GENERATING COMMANDS FOR SEPARATE TERMINALS"
    echo "=================================================================================="
    echo ""
    echo "Run each of these commands in a separate terminal window:"
    echo ""
    
    for answerer in "${ANSWER_MODELS[@]}"; do
        local cmd="python3 ${STAGE3_SCRIPT} --config ${CONFIG}"
        if [[ -n "${DATASET}" ]]; then
            cmd="${cmd} --dataset ${DATASET}"
        fi
        cmd="${cmd} --answerer ${answerer} --skip-dynamic"
        if [[ -n "${SKIP_EXISTING}" ]]; then
            cmd="${cmd} ${SKIP_EXISTING}"
        fi
        if [[ -n "${DRY_RUN}" ]]; then
            cmd="${cmd} ${DRY_RUN}"
        fi
        if [[ ${#OTHER_ARGS[@]} -gt 0 ]]; then
            cmd="${cmd} ${OTHER_ARGS[*]}"
        fi
        echo "${cmd}"
        echo ""
    done
    
    echo "=================================================================================="
    echo "After all answerers complete, run dynamic quality pass:"
    local dyn_cmd="python3 ${STAGE3_SCRIPT} --config ${CONFIG}"
    if [[ -n "${DATASET}" ]]; then
        dyn_cmd="${dyn_cmd} --dataset ${DATASET}"
    fi
    dyn_cmd="${dyn_cmd} --skip-existing"
    echo "${dyn_cmd}"
    echo "=================================================================================="
    exit 0
fi

# Run in parallel using background processes
echo "Running answerers in parallel (background processes)..."
echo "Log files: stage3_<answerer>.log"
echo ""

PIDS=()
FAILED=()

for answerer in "${ANSWER_MODELS[@]}"; do
    run_one_answerer "${answerer}" &
    PIDS+=($!)
    echo "  Started ${answerer} (PID: $!)"
    # Small delay to avoid overwhelming the system
    sleep 2
done

echo ""
echo "All answerers started. Waiting for completion..."
echo ""

# Wait for all processes and collect exit codes
FAILED_COUNT=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    answerer="${ANSWER_MODELS[$i]}"
    if wait "$pid"; then
        echo "✓ ${answerer} completed successfully"
    else
        echo "✗ ${answerer} failed (PID: $pid)"
        FAILED+=("${answerer}")
        ((FAILED_COUNT++))
    fi
done

echo ""
echo "=================================================================================="
echo "Answering phase completed"
echo "=================================================================================="
echo "Successful: $((${#ANSWER_MODELS[@]} - FAILED_COUNT))/${#ANSWER_MODELS[@]}"
if [[ ${FAILED_COUNT} -gt 0 ]]; then
    echo "Failed: ${FAILED_COUNT}"
    echo "  ${FAILED[*]}"
    echo ""
    echo "Check log files: stage3_<answerer>.log"
fi
echo ""

# Run dynamic quality pass if not skipped
if [[ -z "${SKIP_DYNAMIC}" ]]; then
    echo "=================================================================================="
    echo "Running dynamic quality pass..."
    echo "=================================================================================="
    echo "Note: This will process all answerer results together for quality analysis"
    echo ""
    
    local dyn_cmd=(
        "${STAGE3_SCRIPT}"
        --config "${CONFIG}"
    )
    
    if [[ -n "${DATASET}" ]]; then
        dyn_cmd+=(--dataset "${DATASET}")
    fi
    
    dyn_cmd+=(--skip-existing)
    # Don't add --skip-dynamic here - we want to run dynamic quality
    
    python3 "${dyn_cmd[@]}"
    
    echo ""
    echo "✓ Dynamic quality pass completed"
else
    echo "Skipping dynamic quality pass (--skip-dynamic was set)"
    echo ""
    echo "To run it manually later:"
    local dyn_cmd="python3 ${STAGE3_SCRIPT} --config ${CONFIG}"
    if [[ -n "${DATASET}" ]]; then
        dyn_cmd="${dyn_cmd} --dataset ${DATASET}"
    fi
    dyn_cmd="${dyn_cmd} --skip-existing"
    echo "  ${dyn_cmd}"
fi

echo ""
echo "=================================================================================="
if [[ ${FAILED_COUNT} -eq 0 ]]; then
    echo "✓ All done!"
    exit 0
else
    echo "⚠ Completed with ${FAILED_COUNT} failure(s)"
    exit 1
fi

