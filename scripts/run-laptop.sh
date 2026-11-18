#!/usr/bin/env bash
# Run change-detect on a Mac station (isr-hq). Reads configuration from the ENVIRONMENT only
# (the platform's laptop/start-mac.sh sources ~/.config/isr-kit/station.env first).
# Runs in the foreground on 0.0.0.0:${WEB_PORT:-8130} until SIGTERM/SIGINT, then exits cleanly
# (publishes a retained "stopped" heartbeat).
# Usage: scripts/run-laptop.sh [--simulate] [extra changedetect args]
#        scripts/run-laptop.sh --setup-only [--export-onnx]
#   --setup-only: install dependencies, download every model needed offline (YOLO weights, and the
#   Ollama VLM when it runs on this Mac), then exit. Run it once while online, before the hotspot.
#   --export-onnx (with --setup-only): also write yolo11n.onnx for the Pi.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
VENV="${CHANGE_DETECT_VENV:-$REPO/.venv-laptop}"

# Mac defaults from the platform contract; anything already in the environment wins.
export TAK_HOST="${TAK_HOST:-isr-pi.local}"
export TAK_PORT="${TAK_PORT:-8087}"
export MQTT_HOST="${MQTT_HOST:-isr-pi.local}"
export MQTT_PORT="${MQTT_PORT:-1883}"
export MQTT_USER="${MQTT_USER:-isrkit}"
export MQTT_PASS="${MQTT_PASS:-}"
export WEB_PORT="${WEB_PORT:-8130}"
export VLM_URL="${VLM_URL:-http://localhost:11434}"
export VLM_MODEL="${VLM_MODEL:-qwen3-vl:8b-instruct}"
export CHANGE_DETECT_PROFILE="${CHANGE_DETECT_PROFILE:-laptop}"
export CHANGE_DETECT_DATA_DIR="${CHANGE_DETECT_DATA_DIR:-$HOME/change-detect-data}"
export PYTHONUNBUFFERED=1

log() { echo "[change-detect] $*" >&2; }

SETUP_ONLY=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--setup-only" ]; then SETUP_ONLY=1; else ARGS+=("$a"); fi
done

if [ ! -x "$VENV/bin/python" ]; then
    log "creating $VENV"
    if command -v uv >/dev/null 2>&1; then
        uv venv -q -p 3.12 "$VENV"
    else
        PY=""
        for c in python3.12 python3.11 python3.13 python3; do
            if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY="$c"; break; fi
        done
        [ -n "$PY" ] || { log "need Python 3.11+ (brew install python@3.12, or install uv)"; exit 1; }
        "$PY" -m venv "$VENV"
    fi
fi
STAMP="$VENV/.deps-$(cat requirements.txt requirements-desktop.txt | shasum | cut -c1-12)"
if [ ! -f "$STAMP" ]; then
    log "installing dependencies (first run downloads PyTorch, ~1 GB)"
    if command -v uv >/dev/null 2>&1; then
        uv pip install -q -p "$VENV/bin/python" -r requirements-desktop.txt
    else
        "$VENV/bin/python" -m pip install -q --upgrade pip
        "$VENV/bin/python" -m pip install -q -r requirements-desktop.txt
    fi
    touch "$STAMP"
fi

if [ "$SETUP_ONLY" = 1 ]; then
    "$VENV/bin/python" -m changedetect.prefetch --data-dir "$CHANGE_DETECT_DATA_DIR" ${ARGS[@]+"${ARGS[@]}"}
    if [ "${VLM_BACKEND:-auto}" != "off" ]; then
        case "$VLM_URL" in
            *://localhost*|*://127.0.0.1*)
                if command -v ollama >/dev/null 2>&1; then
                    log "pulling VLM $VLM_MODEL with Ollama (several GB the first time)"
                    ollama pull "$VLM_MODEL" || log "WARNING: 'ollama pull $VLM_MODEL' failed; AI suggestions will be off offline"
                else
                    log "Ollama not installed: AI suggestions need it (brew install ollama; ollama pull $VLM_MODEL)"
                fi ;;
            *) log "VLM is remote ($VLM_URL): run 'ollama pull $VLM_MODEL' on that host" ;;
        esac
    fi
    log "setup done; models are in $CHANGE_DETECT_DATA_DIR/models"
    exit 0
fi

[ -n "$MQTT_PASS" ] || log "WARNING: MQTT_PASS is empty; heartbeat/mesh to $MQTT_HOST will be refused"
if [ "${VLM_BACKEND:-auto}" != "off" ] && command -v curl >/dev/null 2>&1; then
    if curl -fsS -m 2 "${VLM_URL%/}/v1/models" >/dev/null 2>&1; then
        log "local VLM up at $VLM_URL (model $VLM_MODEL)"
    else
        log "no VLM at $VLM_URL: on isr-hq run 'ollama pull $VLM_MODEL' for offline AI suggestions"
    fi
fi
log "TAK $TAK_HOST:$TAK_PORT | MQTT $MQTT_USER@$MQTT_HOST:$MQTT_PORT | port $WEB_PORT | data $CHANGE_DETECT_DATA_DIR"

# Keep the Mac awake for as long as this process lives (caffeinate -w follows our PID, which
# exec hands to Python, so SIGTERM goes straight to Python and caffeinate exits with it).
if [ "$(uname)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1; then
    caffeinate -is -w $$ &
fi
exec "$VENV/bin/python" -m changedetect --host 0.0.0.0 --port "$WEB_PORT" --profile "$CHANGE_DETECT_PROFILE" ${ARGS[@]+"${ARGS[@]}"}
