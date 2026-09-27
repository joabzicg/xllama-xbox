#!/usr/bin/env bash
# Run against the Xbox LAN endpoint. Requires curl and Python 3; no model runs here.
# Example: ./scripts/lan-stream-validate.sh --base-url http://192.168.1.50:11434 --model qwen25-vl-3b
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/lan_validate.py" stream "$@"
