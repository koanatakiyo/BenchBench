#!/usr/bin/env bash
# Helper to answer every csbench Stage-3 question with every configured model.
# By default it reads benchbench/config/stage3_csbench.yaml and runs the complete
# designer × answerer matrix via run_stage3_csbench.py.
#
# Usage:
#   ./run_all_models.sh                        # run with default config
#   ./run_all_models.sh /path/to/config.yaml   # use custom config
#   ./run_all_models.sh "" --dry-run --limit 8 # forward extra args
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_stage3_csbench.py"
DEFAULT_CONFIG="${PROJECT_ROOT}/config/stage3_csbench.yaml"

CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"
shift || true

echo "=== Stage 3 Answerer Panel ==="
echo "Config: ${CONFIG_PATH}"
python "${RUNNER}" --config "${CONFIG_PATH}" "$@"

echo "Stage 3 answerer runs complete."


