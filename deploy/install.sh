#!/usr/bin/env bash
# Install change-detect on the Pi 5. Idempotent; no prompts; does NOT enable at boot
# (pi-profiles starts/stops it).
# Usage: sudo bash deploy/install.sh
set -euo pipefail

APP=change-detect
APP_DIR=/opt/$APP
DATA_DIR=/var/lib/$APP
ENV_FILE=/etc/$APP.env
SVC_USER=changedetect
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo bash deploy/install.sh" >&2
    exit 1
fi

echo "=== $APP install ==="

echo "[1/6] System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip libglib2.0-0 rsync >/dev/null

echo "[2/6] Service user"
if ! id -u "$SVC_USER" >/dev/null 2>&1; then
    useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin "$SVC_USER"
fi

echo "[3/6] Copy code to $APP_DIR"
mkdir -p "$APP_DIR"
if [ "$(realpath "$SRC_DIR")" = "$(realpath "$APP_DIR")" ]; then
    echo "    running from $APP_DIR itself: skipping copy"
else
    rsync -a --delete --exclude '.venv' --exclude '__pycache__' --exclude 'data' \
        "$SRC_DIR/changedetect" "$SRC_DIR/deploy" "$SRC_DIR/profiles.d" "$SRC_DIR/requirements.txt" "$APP_DIR/"
fi
chmod +x "$APP_DIR/deploy/start.sh"

echo "[4/6] Python venv"
if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
    python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

echo "[5/6] Data dir and config"
mkdir -p "$DATA_DIR"/{reports,uploads,samples,models}
echo "    model weights for offline use"
# YOLO is optional: a ready yolo11n.onnx in the repo's models/ (exported on a Mac with
# python -m changedetect.prefetch --export-onnx) or YOLO_ONNX_URL in the env file is copied in.
( cd "$APP_DIR"; set -a; [ -f "$ENV_FILE" ] && . "$ENV_FILE"; set +a
  "$APP_DIR/.venv/bin/python" -m changedetect.prefetch --data-dir "$DATA_DIR" \
      --from "$SRC_DIR/models" --from "$DATA_DIR/models" ) || echo "    WARNING: model prefetch failed (YOLO labels may be off)"
chown -R "$SVC_USER:$SVC_USER" "$DATA_DIR"
if [ ! -f "$ENV_FILE" ]; then
    install -m 600 -o root -g root "$SRC_DIR/deploy/$APP.env.example" "$ENV_FILE"
    echo "    created $ENV_FILE (edit it to add GEMINI_API_KEY etc.)"
else
    echo "    keeping existing $ENV_FILE"
fi

echo "[6/6] systemd unit (not enabled at boot)"
install -m 644 "$APP_DIR/deploy/$APP.service" /etc/systemd/system/$APP.service
systemctl daemon-reload

if [ -d /opt/pi-profiles/profiles.d ]; then
    echo "pi-profiles found: installing profile configs"
    cp "$SRC_DIR"/profiles.d/*.conf /opt/pi-profiles/profiles.d/
fi

cat <<MSG

=== Install complete ===
  Web UI:    http://<pi-ip>:8130
  Start:     sudo pi-profiles $APP      (or: sudo systemctl start $APP)
  Logs:      journalctl -u $APP -f
  Config:    $ENV_FILE
  Reports:   $DATA_DIR/reports
  YOLO:      optional: put yolo11n.onnx in the repo's models/ and re-run, or see README
MSG
