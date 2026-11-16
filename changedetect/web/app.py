"""FastAPI web UI and JSON API (port 8130)."""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from changedetect import __version__, cot, media, pipeline, synthetic
from changedetect.config import Settings
from changedetect.gemini import GeminiAdvisor
from changedetect.labeler import Labeler
from changedetect.mesh import MeshBus

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
_ID_RE = re.compile(r"^[0-9A-Za-z-]{6,40}$")


class CotRequest(BaseModel):
    lat: float | None = None
    lon: float | None = None
    callsign: str | None = None


class MeshRequest(BaseModel):
    to: str | None = None
    text: str | None = None


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.labeler = Labeler(settings.yolo_model)
        self.gemini = GeminiAdvisor(settings.gemini_api_key, settings.gemini_model)
        self.cot = cot.CotSender(settings.tak_host, settings.tak_port, settings.cot_enabled)
        self.mesh = MeshBus(settings.mqtt_host, settings.mqtt_port, settings.mqtt_user,
                            settings.mqtt_pass, settings.mesh_enabled)
        self.lock = threading.Lock()  # one analysis at a time (CPU-bound on the Pi)
        self.samples: list[dict] = []


def create_app(settings: Settings) -> FastAPI:
    settings.ensure_dirs()
    svc = Services(settings)
    svc.samples = synthetic.write_samples(settings.samples_dir)
    app = FastAPI(title="Change Detect", version=__version__)
    app.state.svc = svc
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/results", StaticFiles(directory=str(settings.reports_dir)), name="results")

    def _result_dir(rid: str) -> Path:
        if not _ID_RE.match(rid):
            raise HTTPException(404, "unknown result")
        d = settings.reports_dir / rid
        if not (d / "result.json").exists():
            raise HTTPException(404, "unknown result")
        return d

    def _load(rid: str) -> tuple[Path, dict]:
        d = _result_dir(rid)
        return d, json.loads((d / "result.json").read_text())

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/api/status")
    async def status():
        return {
            "version": __version__,
            "simulate": settings.simulate,
            "gemini": svc.gemini.status(),
            "yolo": {"enabled": svc.labeler.enabled, "model": str(settings.yolo_model) if settings.yolo_model else None,
                     "error": None if svc.labeler.enabled else svc.labeler.error},
            "tak": svc.cot.status(),
            "mesh": svc.mesh.status(),
            "busy": svc.lock.locked(),
        }

    @app.get("/api/samples")
    async def samples():
        return [{k: s[k] for k in ("id", "title", "before", "after")} for s in svc.samples]

    @app.get("/api/results")
    async def list_results(limit: int = 30):
        out = []
        dirs = sorted((p for p in settings.reports_dir.iterdir() if (p / "result.json").exists()), reverse=True)
        for d in dirs[:limit]:
            try:
                r = json.loads((d / "result.json").read_text())
            except ValueError:
                continue
            out.append({"id": r["id"], "created": r["created"], "summary": r["summary"], "mode": r["mode"],
                        "changes_total": r["changes_total"], "alignment_ok": r["alignment_ok"]})
        return out

    @app.get("/api/results/{rid}")
    async def get_result(rid: str):
        return _load(rid)[1]

    async def _save_upload(up: UploadFile) -> Path:
        name = Path(up.filename or "upload").name
        try:
            media.kind_of(name)
        except media.MediaError as e:
            raise HTTPException(400, str(e))
        dest = settings.uploads_dir / f"{uuid.uuid4().hex[:8]}_{re.sub(r'[^A-Za-z0-9._-]', '_', name)}"
        limit = settings.max_upload_mb * 1024 * 1024
        size = 0
        with open(dest, "wb") as f:
            while chunk := await up.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
                f.write(chunk)
        return dest

    @app.post("/api/analyze")
    async def analyze(
        before: UploadFile | None = File(None), after: UploadFile | None = File(None),
        sample: str | None = Form(None), sensitivity: float = Form(0.5), min_area: int = Form(300),
        mode360: str = Form("auto"), use_yolo: bool = Form(True),
        before_time_s: float | None = Form(None), after_time_s: float | None = Form(None),
    ):
        tmp: list[Path] = []
        try:
            if sample:
                s = next((x for x in svc.samples if x["id"] == sample), None)
                if s is None:
                    raise HTTPException(404, "unknown sample")
                bpath, apath = settings.samples_dir / s["before"], settings.samples_dir / s["after"]
            else:
                if before is None or after is None:
                    raise HTTPException(400, "upload both a BEFORE and an AFTER file, or pick a sample")
                bpath = await _save_upload(before)
                tmp.append(bpath)
                apath = await _save_upload(after)
                tmp.append(apath)
            if mode360 not in ("auto", "on", "off"):
                raise HTTPException(400, "mode360 must be auto, on or off")
            opts = pipeline.Options(sensitivity=min(max(sensitivity, 0.0), 1.0), min_area=max(20, min_area),
                                    mode360=mode360, use_yolo=use_yolo,
                                    before_time_s=before_time_s, after_time_s=after_time_s)

            def run():
                with svc.lock:
                    return pipeline.analyze(bpath, apath, settings.reports_dir, opts, svc.labeler)

            try:
                result = await run_in_threadpool(run)
            except media.MediaError as e:
                raise HTTPException(400, str(e))
            # Keep the original files with the result for the record (originals keep their EXIF)
            out = settings.reports_dir / result["id"]
            for p, tag in ((bpath, "before"), (apath, "after")):
                try:
                    shutil.copy2(p, out / f"original_{tag}{p.suffix.lower()}")
                except OSError:
                    pass
            return result
        finally:
            for p in tmp:
                p.unlink(missing_ok=True)

    @app.post("/api/results/{rid}/ai")
    async def ai_suggest(rid: str):
        if not svc.gemini.enabled:
            raise HTTPException(400, "Gemini is not configured (offline mode). Set GEMINI_API_KEY to enable.")
        d, result = _load(rid)

        def run():
            n = 0
            for v in result["views"]:
                for ch in v["changes"]:
                    b = (d / ch["crops"]["before"]).read_bytes()
                    a = (d / ch["crops"]["after"]).read_bytes()
                    text = svc.gemini.describe(b, a, ch.get("label_text") or ch["kind"])
                    if text:
                        ch["ai_suggestion"] = text
                        n += 1
            result["ai_model"] = svc.gemini.model
            pipeline.rebuild_report(d, result)
            return n

        n = await run_in_threadpool(run)
        return {"described": n, "error": svc.gemini.last_error, "result": result}

    @app.post("/api/results/{rid}/cot")
    async def push_cot(rid: str, req: CotRequest):
        _, result = _load(rid)
        manual = req.lat is not None or req.lon is not None
        if manual:
            if req.lat is None or req.lon is None or not (-90 <= req.lat <= 90) or not (-180 <= req.lon <= 180):
                raise HTTPException(400, "enter a valid latitude and longitude")
            lat, lon = req.lat, req.lon
        elif result.get("gps"):
            lat, lon = result["gps"]["lat"], result["gps"]["lon"]
        else:
            raise HTTPException(400, "no GPS in these images; enter lat/lon manually")
        if not result["alignment_ok"]:
            raise HTTPException(409, "alignment failed; refusing to report a change")
        callsign = req.callsign or f"CHANGE {result['id'][-4:]} ({result['changes_total']})"
        remarks = f"Change detected: {result['summary']} [change-detect {result['id']}; position from " \
                  f"{'manual entry' if manual else result['gps']['source']}; camera position, not the change itself]"
        xml = cot.build_change_cot(f"change-detect-{result['id']}", lat, lon, callsign, remarks, manual=manual)
        state = svc.cot.send(xml)
        return {"state": state, "tak": svc.cot.status(), "lat": lat, "lon": lon}

    @app.post("/api/results/{rid}/mesh")
    async def push_mesh(rid: str, req: MeshRequest):
        _, result = _load(rid)
        text = req.text or f"CHANGE {result['id'][-4:]}: {result['summary']}"
        if result.get("gps") and not req.text:
            text += f" @{result['gps']['lat']:.5f},{result['gps']['lon']:.5f}"
        ok = await run_in_threadpool(svc.mesh.send, text, req.to)
        return {"sent": ok, "error": svc.mesh.last_error, "text": text[:200]}

    return app
