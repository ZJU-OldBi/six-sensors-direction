#!/usr/bin/env bash
# Headless gas-direction detection (no display required).
# All arguments are forwarded to direction_cli.py, for example:
#   ./run_cli.sh --list-ports
#   ./run_cli.sh --port /dev/ttyUSB0
#   ./run_cli.sh --port /dev/ttyUSB0 --once
#   ./run_cli.sh --port /dev/ttyUSB0 --robot-output udp:192.168.1.20:9000
#   ./run_cli.sh --replay samples/180度/sample_180deg.csv --speed 0
#   ./run_cli.sh --self-test --data-dir samples
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -x "${VENV_DIR}/bin/python" ] || [ ! -f "${VENV_DIR}/.deps_ok" ]; then
    echo "[run_cli] environment not ready; running setup.sh first"
    "${SCRIPT_DIR}/setup.sh"
fi

exec "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/direction_cli.py" "$@"
