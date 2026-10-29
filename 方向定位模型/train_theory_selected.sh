#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 train_theory_selected_model.py --data-dir "../方向传感数据" --output-dir models
