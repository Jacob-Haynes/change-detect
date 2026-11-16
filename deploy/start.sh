#!/usr/bin/env bash
set -euo pipefail
cd /opt/change-detect
exec /opt/change-detect/.venv/bin/python -m changedetect --port "${WEB_PORT:-8130}" ${CHANGE_DETECT_ARGS:-}
