#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 sequence_eight_class_classifier.py train --data-dir "../方向传感数据" --output-dir models
