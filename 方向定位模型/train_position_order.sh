#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 train_direction_models.py --data-dir "../方向传感数据" --output-dir models --method position_order
