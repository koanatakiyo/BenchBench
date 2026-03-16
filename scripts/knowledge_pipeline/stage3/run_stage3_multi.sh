#!/usr/bin/env bash
# Run Stage 3 for a set of datasets with a chosen designer, skipping dynamic pass.
# Usage:
#   DATASETS="wemath_stage2_textonly tombench_en" DESIGNER=llama4_maverick bash run_stage3_multi.sh
# Defaults:
#   DATASETS="medxpertqa_mm_stage2_visualprimed wemath_stage2_textonly wemath_stage2_visualprimed tombench_cn tombench_en"
#   DESIGNER="llama4_maverick"
#   CONFIG="../../../config/stage3.yaml"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../../../config/stage3.yaml}"
DESIGNER="${DESIGNER:-llama4_maverick}"
DATASETS="${DATASETS:-medxpertqa_mm_stage2_visualprimed wemath_stage2_textonly wemath_stage2_visualprimed tombench_cn tombench_en medxpertqa_text}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "Config: ${CONFIG}"
echo "Designer: ${DESIGNER}"
echo "Datasets: ${DATASETS}"
echo "Log level: ${LOG_LEVEL}"

for ds in ${DATASETS}; do
  echo "== Running dataset=${ds} designer=${DESIGNER} =="
  python "${SCRIPT_DIR}/run_stage3.py" \
    --dataset "${ds}" \
    --config "${CONFIG}" \
    --designer "${DESIGNER}" \
    --skip-dynamic \
    --skip-exist \
    --log-level "${LOG_LEVEL}"
done

echo "Done."

