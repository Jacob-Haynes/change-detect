"""Runtime settings, read from the environment (/etc/change-detect.env on the Pi)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("/var/lib/change-detect")
# Current stable Gemini Flash model per Google's model list (Nov 2026). Override with GEMINI_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_PORT = 8130
# Ollama vision model on the HQ Mac (Qwen3-VL is in Ollama's vision library). Override with VLM_MODEL.
DEFAULT_VLM_MODEL = "qwen3-vl:8b"
# Bare Ollama base URL (no path); the client appends /v1/... itself.
DEFAULT_VLM_URL = "http://localhost:11434"

# Processing resolution per deployment profile: the Pi downscales for speed, the laptop keeps
# full-resolution stills (alignment itself always runs on a <=2000 px copy).
PROFILES = {
    "pi": {"max_dim": 1600, "max_dim_360": 4096, "batch_max_files": 120},
    "laptop": {"max_dim": None, "max_dim_360": None, "batch_max_files": 600},
}


def parse_position(lat: str | float | None, lon: str | float | None) -> tuple[float, float] | None:
    """A usable lat/lon pair, or None. 0,0 (and anything out of range or unparsable) is treated as
    unset, so a missing position can never be emitted as 0,0 (contract C4.2)."""
    try:
        la, lo = float(lat), float(lon)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (-90 <= la <= 90 and -180 <= lo <= 180) or (la == 0 and lo == 0):
        return None
    if la != la or lo != lo:  # NaN
        return None
    return la, lo


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _writable(path: Path) -> bool:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return os.access(probe, os.W_OK)


@dataclass
class Settings:
    data_dir: Path
    simulate: bool = False
    tak_host: str = "127.0.0.1"
    tak_port: int = 8087
    cot_enabled: bool = True
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_user: str = "guest"
    mqtt_pass: str = "guest"
    mesh_enabled: bool = True
    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    yolo_model: Path | None = None
    yolo_backend: str = "auto"  # auto | ultralytics | opencv | off
    yolo_weights: str | None = None
    vlm_backend: str = "auto"  # auto (openai_compat -> gemini -> off) | openai_compat | gemini | off
    vlm_url: str = DEFAULT_VLM_URL
    vlm_model: str = DEFAULT_VLM_MODEL
    vlm_api_key: str | None = None
    profile: str = "pi"
    port: int = DEFAULT_PORT
    public_url: str | None = None
    heartbeat_enabled: bool = True
    max_upload_mb: int = 1024
    sensor_position: tuple[float, float] | None = None  # SENSOR_LAT/SENSOR_LON (OP position)

    @classmethod
    def from_env(cls, data_dir: str | Path | None = None, simulate: bool = False,
                 profile: str | None = None) -> "Settings":
        chosen = Path(data_dir or _env("CHANGE_DETECT_DATA_DIR") or DEFAULT_DATA_DIR)
        if not _writable(chosen):
            fallback = Path.cwd() / "data"
            logger.warning("Data dir %s not writable; using %s", chosen, fallback)
            chosen = fallback
        yolo = _env("YOLO_MODEL")
        yolo_path = Path(yolo) if yolo else chosen / "models" / "yolo11n.onnx"
        return cls(
            data_dir=chosen,
            simulate=simulate,
            tak_host=_env("TAK_HOST", "127.0.0.1"),
            tak_port=int(_env("TAK_PORT", "8087")),
            cot_enabled=_env("COT_ENABLED", "1") not in ("0", "false", "no"),
            mqtt_host=_env("MQTT_HOST", "127.0.0.1"),
            mqtt_port=int(_env("MQTT_PORT", "1883")),
            mqtt_user=_env("MQTT_USER", "guest"),
            mqtt_pass=_env("MQTT_PASS", "guest"),
            mesh_enabled=_env("MESH_ENABLED", "1") not in ("0", "false", "no"),
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            yolo_model=yolo_path,
            yolo_backend=_env("YOLO_BACKEND", "auto"),
            yolo_weights=_env("YOLO_WEIGHTS"),
            vlm_backend=_env("VLM_BACKEND", "auto"),
            vlm_url=_env("VLM_URL", DEFAULT_VLM_URL),
            vlm_model=_env("VLM_MODEL", DEFAULT_VLM_MODEL),
            vlm_api_key=_env("VLM_API_KEY"),
            profile=(profile or _env("CHANGE_DETECT_PROFILE", "pi")),
            port=int(_env("WEB_PORT", str(DEFAULT_PORT))),
            public_url=_env("PUBLIC_URL"),
            sensor_position=parse_position(_env("SENSOR_LAT"), _env("SENSOR_LON")),
            heartbeat_enabled=_env("HEARTBEAT", "1") not in ("0", "false", "no"),
            max_upload_mb=int(_env("MAX_UPLOAD_MB", "1024")),
        )

    @property
    def limits(self) -> dict:
        return PROFILES.get(self.profile, PROFILES["pi"])

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.reports_dir, self.uploads_dir, self.samples_dir, self.data_dir / "models",
                  self.data_dir / "batches"):
            d.mkdir(parents=True, exist_ok=True)
