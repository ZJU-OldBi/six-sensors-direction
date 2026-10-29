#!/usr/bin/env bash
# Prepare the Python environment for the Linux gas-direction toolset.
# Creates a local virtual environment (.venv) and installs requirements.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[setup] python3 is required. Example (Ubuntu/Debian):" >&2
    echo "        sudo apt update && sudo apt install -y python3 python3-venv python3-tk" >&2
    exit 1
fi

if ! python3 -c "import tkinter" >/dev/null 2>&1; then
    echo "[setup] warning: python3-tk is missing; the GUI (run_gui.sh) will not start." >&2
    echo "        Install it with: sudo apt install -y python3-tk" >&2
    echo "        The headless tool (run_cli.sh) works without it." >&2
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "[setup] creating virtual environment: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

echo "[setup] installing Python dependencies from requirements.txt"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
touch "${VENV_DIR}/.deps_ok"

echo
echo "[setup] done. Quick start:"
echo "  ./run_gui.sh                          # graphical interface (needs a display)"
echo "  ./run_cli.sh --list-ports             # headless: list serial ports"
echo "  ./run_cli.sh --port /dev/ttyUSB0      # headless: live detection"
echo "  ./run_cli.sh --self-test --data-dir samples   # offline check with bundled samples"
echo "  ./train_all.sh --data-dir /path/to/方向传感数据  # retrain models on this machine"
