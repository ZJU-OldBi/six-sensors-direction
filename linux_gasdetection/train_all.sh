#!/usr/bin/env bash
# One-click retraining on Linux, equivalent to the Windows train_all.bat.
#
# Runs the full pipeline:
#   1. data quality screening report
#   2. train all four models (order / sequence_8class / first_response / position_order)
#   3. held-out backtest of the order model
#   4. held-out backtest of the eight-class model
#
# Usage:
#   ./train_all.sh                                # data: ../方向传感数据  (default)
#   ./train_all.sh --data-dir /path/to/data       # custom angle folder tree
#   ./train_all.sh --output-dir models            # custom model output folder
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
DATA_DIR="${SCRIPT_DIR}/../方向传感数据"
OUTPUT_DIR="${SCRIPT_DIR}/models"

while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./train_all.sh [--data-dir DIR] [--output-dir DIR]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ ! -d "${DATA_DIR}" ]; then
    echo "[train] data directory not found: ${DATA_DIR}" >&2
    echo "[train] copy the angle folders (0度, 45度, ...) to that path or pass --data-dir" >&2
    exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ] || [ ! -f "${VENV_DIR}/.deps_ok" ]; then
    echo "[train] environment not ready; running setup.sh first"
    "${SCRIPT_DIR}/setup.sh"
fi

PY="${VENV_DIR}/bin/python"
mkdir -p "${OUTPUT_DIR}"

echo "[train] step 1/4: data quality report"
"${PY}" "${SCRIPT_DIR}/analyze_direction_data.py" --data-dir "${DATA_DIR}" --output "${OUTPUT_DIR}/direction_data_quality.csv"

echo "[train] step 2/4: train all models"
"${PY}" "${SCRIPT_DIR}/train_direction_models.py" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" --method all

echo "[train] step 3/4: backtest order model (held-out cross-validation)"
"${PY}" "${SCRIPT_DIR}/backtest_training_data.py" --data-dir "${DATA_DIR}" \
    --model "${OUTPUT_DIR}/direction_model_order.joblib" --mode cv \
    --output "${OUTPUT_DIR}/training_backtest_cv.csv"

echo "[train] step 4/4: backtest eight-class model (held-out cross-validation)"
"${PY}" "${SCRIPT_DIR}/backtest_training_data.py" --data-dir "${DATA_DIR}" \
    --model "${OUTPUT_DIR}/direction_model_sequence_8class.joblib" --method sequence_8class --mode cv \
    --output "${OUTPUT_DIR}/sequence_8class_backtest_cv.csv"

echo
echo "[train] done. Models and reports are in: ${OUTPUT_DIR}"
