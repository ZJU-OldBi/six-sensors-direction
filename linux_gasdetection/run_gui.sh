#!/usr/bin/env bash
# Start the gas-direction graphical interface (Tk).
# All arguments are forwarded to direction_inference.py, for example:
#   ./run_gui.sh --port /dev/ttyUSB0
#   ./run_gui.sh --replay samples/0度/sample_0deg.csv
#   ./run_gui.sh --robot-output udp:192.168.1.20:9000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -x "${VENV_DIR}/bin/python" ] || [ ! -f "${VENV_DIR}/.deps_ok" ]; then
    echo "[run_gui] environment not ready; running setup.sh first"
    "${SCRIPT_DIR}/setup.sh"
fi

exec "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/direction_inference.py" "$@"
