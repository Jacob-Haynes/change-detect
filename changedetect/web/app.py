"""FastAPI web UI and JSON API (port 8130)."""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from changedetect import __version__, batch, cot, media, pipeline, synthetic
from changedetect.config import Settings
from changedetect.gemini import GeminiAdvisor
from changedetect.heartbeat import Heartbeat, short_hostname
from changedetect.labeler import detect_accelerator, make_labeler
from changedetect.config import parse_position
from changedetect.mesh import MeshBus, label_simulated
from changedetect.tak_client import TakClient
from changedetect.vlm import AdvisorChain, OpenAICompatAdvisor

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
        self.labeler = make_labeler(settings.yolo_backend, settings.yolo_model, settings.data_dir / "models",
                                    settings.yolo_weights)
        self.gemini = GeminiAdvisor(settings.gemini_api_key, settings.gemini_model)
        self.advisor = AdvisorChain(settings.vlm_backend,
                                    OpenAICompatAdvisor(settings.vlm_url, settings.vlm_model, settings.vlm_api_key),
                                    self.gemini)
        self.accelerator = getattr(self.labeler, "device", None) or detect_accelerator()
        # C5.1: one long-lived, registered connection to OpenTAKServer, opened at startup
        self.cot = TakClient(settings.tak_host, settings.tak_port, short_hostname(), settings.sensor_position,
                             __version__, settings.cot_enabled)
        self.mesh = MeshBus(settings.mqtt_host, settings.mqtt_port, settings.mqtt_user,
                            settings.mqtt_pass, settings.mesh_enabled)
        self.lock = threading.Lock()  # one analysis at a time (CPU-bound on the Pi)
        self.samples: list[dict] = []
        self.batches = batch.BatchRunner(settings.data_dir / "batches", settings.reports_dir, self.lock, self.labeler)
        self.heartbeat = Heartbeat(settings.mqtt_host, settings.mqtt_port, settings.mqtt_user, settings.mqtt_pass,
                                   settings.port, mode_fn=self.mode, public_url=settings.public_url,
                                   status_fn=self.health,
                                   enabled=settings.heartbeat_enabled)

    def health(self) -> str:
        """ok | degraded (an integration is failing: TAK markers stuck, mesh send failed)."""
        tak = self.cot.status()
        if (tak["enabled"] and tak["pending"] and not tak["connected"]) or self.mesh.last_error:
            return "degraded"
        return "ok"

    def mode(self) -> str:
        if self.settings.simulate:
            return "simulate"
        return self.accelerator if self.accelerator in ("cuda", "mps") else "cpu"

    def options(self, **kw) -> pipeline.Options:
        lim = self.settings.limits
        return pipeline.Options(max_dim=lim["max_dim"], max_dim_360=lim["max_dim_360"], **kw)


def create_app(settings: Settings) -> FastAPI:
    settings.ensure_dirs()
    svc = Services(settings)
    svc.samples = synthetic.write_samples(settings.samples_dir)
    svc.heartbeat.start()
    app = FastAPI(title="Change Detect", version=__version__)
    app.state.svc = svc
    svc.cot.start()
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
            "profile": settings.profile,
            "mode": svc.mode(),
            "accelerator": svc.accelerator,
            "max_dim": settings.limits["max_dim"],
            "reports_dir": str(settings.reports_dir),
            "gemini": svc.gemini.status(),
            "ai": svc.advisor.status(),
            "yolo": {"enabled": svc.labeler.enabled, "device": getattr(svc.labeler, "device", "cpu"),
                     "model": str(svc.labeler.model_path) if svc.labeler.model_path else None,
                     "error": None if svc.labeler.enabled else svc.labeler.error},
            "heartbeat": svc.heartbeat.status(),
            "tak": svc.cot.status(),
            "mesh": svc.mesh.status(),
            "sensor": ({"lat": settings.sensor_position[0], "lon": settings.sensor_position[1]}
                       if settings.sensor_position else None),
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
            opts = svc.options(sensitivity=min(max(sensitivity, 0.0), 1.0), min_area=max(20, min_area),
                               mode360=mode360, use_yolo=use_yolo,
                               before_time_s=before_time_s, after_time_s=after_time_s)

            def run():
                with svc.lock:
                    return pipeline.analyze(bpath, apath, settings.reports_dir, opts, svc.labeler)

            try:
                result = await run_in_threadpool(run)
            except media.MediaError as e:
                raise HTTPException(400, str(e))
            if sample and not result.get("synthetic"):
                result["synthetic"] = True
                pipeline.save_result(settings.reports_dir / result["id"], result)
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
        if not svc.advisor.enabled:
            raise HTTPException(400, "AI suggestions are off: no local VLM reachable "
                                     f"({svc.advisor.local.url}, model {svc.advisor.local.model}) and no GEMINI_API_KEY.")
        d, result = _load(rid)

        def run():
            n = 0
            for v in result["views"]:
                for ch in v["changes"]:
                    b = (d / ch["crops"]["before"]).read_bytes()
                    a = (d / ch["crops"]["after"]).read_bytes()
                    text, source = svc.advisor.describe(b, a, ch.get("label_text") or ch["kind"])
                    if text:
                        ch["ai_suggestion"] = text
                        ch["ai_source"] = source
                        n += 1
            result["ai_model"] = svc.advisor.source_label(svc.advisor.last_source)
            pipeline.rebuild_report(d, result)
            return n

        n = await run_in_threadpool(run)
        return {"described": n, "error": svc.advisor.last_error if n == 0 else None, "result": result}

    @app.post("/api/results/{rid}/cot")
    async def push_cot(rid: str, req: CotRequest):
        _, result = _load(rid)
        manual = req.lat is not None or req.lon is not None
        if manual:
            pos = parse_position(req.lat, req.lon)
            if pos is None:
                raise HTTPException(400, "enter a valid latitude and longitude (0,0 is not accepted)")
            lat, lon = pos
            source = "manual entry"
        elif result.get("gps"):
            lat, lon = result["gps"]["lat"], result["gps"]["lon"]
            source = result["gps"]["source"]
        elif settings.sensor_position:
            lat, lon = settings.sensor_position
            source = "SENSOR_LAT/SENSOR_LON (OP position)"
        else:
            raise HTTPException(400, "no GPS in these images and SENSOR_LAT/SENSOR_LON is not set; "
                                     "enter lat/lon manually")
        if not result["alignment_ok"]:
            raise HTTPException(409, "alignment failed; refusing to report a change")
        uid = f"CHANGE-{result['id']}"
        wait = svc.cot.retry_in(uid)
        if wait > 0:
            raise HTTPException(429, f"marker for this result was just sent; wait {wait:.0f} s")
        callsign = req.callsign or f"CHANGE {result['id'][-4:]} ({result['changes_total']})"
        remarks = f"Change detected: {result['summary']} [change-detect {result['id']}; position from " \
                  f"{source}; camera position, not the change itself]"
        if result.get("synthetic"):
            remarks = "SIMULATED: " + remarks
        xml = cot.build_change_cot(uid, lat, lon, callsign, remarks, manual=manual,
                                   parent_uid=svc.cot.uid, parent_callsign=svc.cot.callsign)
        state = svc.cot.send(xml, uid=uid)
        if state == "rate_limited":
            raise HTTPException(429, "marker for this result was just sent; wait a few seconds")
        return {"state": state, "tak": svc.cot.status(), "lat": lat, "lon": lon, "position_source": source,
                "simulated": bool(result.get("synthetic"))}

    @app.post("/api/results/{rid}/mesh")
    async def push_mesh(rid: str, req: MeshRequest):
        _, result = _load(rid)
        text = req.text or f"CHANGE {result['id'][-4:]}: {result['summary']}"
        if not req.text:
            pos = (result["gps"]["lat"], result["gps"]["lon"]) if result.get("gps") else settings.sensor_position
            if pos:
                text += f" @{pos[0]:.5f},{pos[1]:.5f}"
        text = label_simulated(text, bool(result.get("synthetic")))
        wait = svc.mesh.retry_in()
        if svc.mesh.enabled and wait > 0:
            raise HTTPException(429, f"mesh is limited to one text every 30 s; try again in {wait:.0f} s")
        ok = await run_in_threadpool(svc.mesh.send, text, req.to)
        if not ok and svc.mesh.last_note:
            raise HTTPException(429, svc.mesh.last_note)
        return {"sent": ok, "error": svc.mesh.last_error, "text": text[:200]}

    # ------------------------------------------------------------ folder / SD-card batches

    @app.post("/api/batch")
    async def start_batch(request: Request):
        form = await request.form(max_files=settings.limits["batch_max_files"] * 2 + 10, max_fields=200)
        groups = {k: [f for f in form.getlist(k) if hasattr(f, "filename")] for k in ("files", "before_files", "after_files")}
        try:
            mtimes = json.loads(form.get("mtimes") or "{}")
        except ValueError:
            mtimes = {}
        n = sum(len(v) for v in groups.values())
        if n > settings.limits["batch_max_files"]:
            raise HTTPException(413, f"too many files ({n}); limit is {settings.limits['batch_max_files']} per batch")
        if not groups["files"] and not (groups["before_files"] and groups["after_files"]):
            raise HTTPException(400, "drop one SD-card/DCIM folder, or a BEFORE folder and an AFTER folder")
        bid = svc.batches.new_id()
        updir = settings.uploads_dir / bid
        saved: dict[str, list[Path]] = {}
        limit = settings.max_upload_mb * 1024 * 1024
        for key, files in groups.items():
            sub = updir / key
            sub.mkdir(parents=True, exist_ok=True)
            saved[key] = []
            for i, up in enumerate(files):
                name = Path(up.filename or "f").name
                if name.startswith("."):
                    continue  # macOS ._ resource forks on SD cards
                try:
                    if media.kind_of(name) != "image":
                        continue
                except media.MediaError:
                    continue  # skip .LRV/.THM/.MP4 sidecars etc.
                dest = sub / f"{i:04d}_{re.sub(r'[^A-Za-z0-9._-]', '_', name)}"
                size = 0
                with open(dest, "wb") as f:
                    while chunk := await up.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise HTTPException(413, f"{name} larger than {settings.max_upload_mb} MB")
                        f.write(chunk)
                saved[key].append(dest)
        g = lambda k: form.get(k)
        opts = svc.options(sensitivity=float(g("sensitivity") or 0.5), min_area=max(20, int(float(g("min_area") or 300))),
                           mode360=g("mode360") or "auto", use_yolo=(g("use_yolo") or "true") == "true")
        return svc.batches.start(updir, saved["before_files"], saved["after_files"], saved["files"],
                                 mtimes, opts, bid)

    @app.get("/api/batch/{bid}")
    async def get_batch(bid: str):
        if not _ID_RE.match(bid):
            raise HTTPException(404, "unknown batch")
        job = svc.batches.get(bid)
        if job is None:
            raise HTTPException(404, "unknown batch")
        return job

    @app.get("/api/batches")
    async def list_batches(limit: int = 20):
        return svc.batches.list(limit)

    return app
