#!/usr/bin/env bash
# Run change-detect on a laptop (primary: Apple Silicon Mac; Linux works too).
# Usage: scripts/run-laptop.sh [--simulate] [extra changedetect args]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
VENV="$REPO/.venv-laptop"

if [ ! -f laptop.env ]; then
    cp deploy/laptop.env.example laptop.env
    echo ">> Created laptop.env: set MQTT_PASS (the isrkit password from the Pi) and re-run if needed."
fi
# Load laptop.env, but let variables already set in the shell win
while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    key="${line%%=*}"; val="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [ -z "${!key+x}" ]; then export "$key=$val"; fi
done < laptop.env
export CHANGE_DETECT_DATA_DIR="${CHANGE_DETECT_DATA_DIR:-$HOME/change-detect-data}"
export CHANGE_DETECT_PROFILE="${CHANGE_DETECT_PROFILE:-laptop}"

if [ ! -x "$VENV/bin/python" ]; then
    echo ">> Creating $VENV"
    if command -v uv >/dev/null 2>&1; then
        uv venv -q -p 3.12 "$VENV"
    else
        PY=""
        for c in python3.12 python3.11 python3.13 python3; do
            if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY="$c"; break; fi
        done
        [ -n "$PY" ] || { echo "Need Python 3.11+ (brew install python@3.12, or install uv)"; exit 1; }
        "$PY" -m venv "$VENV"
    fi
fi
STAMP="$VENV/.deps-$(cat requirements.txt requirements-desktop.txt | shasum | cut -c1-12)"
if [ ! -f "$STAMP" ]; then
    echo ">> Installing dependencies (first run downloads PyTorch, ~1 GB)"
    if command -v uv >/dev/null 2>&1; then
        uv pip install -q -p "$VENV/bin/python" -r requirements-desktop.txt
    else
        "$VENV/bin/python" -m pip install -q --upgrade pip
        "$VENV/bin/python" -m pip install -q -r requirements-desktop.txt
    fi
    touch "$STAMP"
fi

ACCEL=$("$VENV/bin/python" -c 'from changedetect.labeler import detect_accelerator as d; print(d())' 2>/dev/null || echo cpu)
echo ">> Accelerator: $ACCEL | TAK $TAK_HOST:$TAK_PORT | MQTT $MQTT_USER@$MQTT_HOST:$MQTT_PORT | data $CHANGE_DETECT_DATA_DIR"
if command -v curl >/dev/null 2>&1 && [ "${VLM_BACKEND:-auto}" != "off" ]; then
    if curl -fsS -m 2 "${VLM_URL%/}/models" >/dev/null 2>&1; then
        echo ">> Local VLM endpoint up at $VLM_URL (model ${VLM_MODEL})"
    else
        echo ">> No local VLM at $VLM_URL: install Ollama and run 'ollama pull ${VLM_MODEL}' for offline AI suggestions"
    fi
fi
if ! "$VENV/bin/python" -c "import socket,sys; socket.gethostbyname(sys.argv[1])" "$MQTT_HOST" 2>/dev/null; then
    echo ">> WARNING: cannot resolve $MQTT_HOST. Join the Pi hotspot, or set the Pi's IP in laptop.env. Continuing."
fi

CMD=("$VENV/bin/python" -m changedetect --profile "$CHANGE_DETECT_PROFILE" "$@")
if [ "$(uname)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -is "${CMD[@]}"   # keep the Mac awake while serving
fi
exec "${CMD[@]}"
