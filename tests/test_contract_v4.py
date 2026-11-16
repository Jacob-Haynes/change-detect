"""Contract v4 (C4.1-C4.4) and rehearsal bugs CD-1 (simulated labelling) and CD-2 (offline models)."""

import io
import json
import socket
import sys
import time
import types

import cv2
import pytest
from fastapi.testclient import TestClient

from changedetect import heartbeat, labeler, mesh, prefetch, tak_client
from changedetect.config import Settings, parse_position
from changedetect.web.app import create_app


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------- C4.1 heartbeat host
def test_heartbeat_host_from_isr_host(monkeypatch):
    monkeypatch.setenv("ISR_HOST", "isr-hq")
    hb = heartbeat.Heartbeat("127.0.0.1", 1, "u", "p", 8130, mode_fn=lambda: "mps", status_fn=lambda: "ok",
                             enabled=False)
    p = hb.payload()
    assert p["host"] == "isr-hq" and p["url"] == "http://isr-hq.local:8130/"


def test_heartbeat_host_defaults_to_short_lower_hostname(monkeypatch):
    monkeypatch.delenv("ISR_HOST", raising=False)
    monkeypatch.setattr(heartbeat.socket, "gethostname", lambda: "ISR-Pi.local")
    assert heartbeat.short_hostname() == "isr-pi"


# ---------------------------------------------------------------- C4.2 sensor position
def test_parse_position_rejects_zero_and_junk():
    assert parse_position("51.15", "-1.75") == (51.15, -1.75)
    assert parse_position(0, 0) is None
    assert parse_position("0", "0.0") is None
    assert parse_position(95, 0) is None
    assert parse_position(None, "1") is None
    assert parse_position("nan", "nan") is None
    assert parse_position("", "") is None


def test_sensor_position_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSOR_LAT", "51.5007")
    monkeypatch.setenv("SENSOR_LON", "-0.1246")
    assert Settings.from_env(tmp_path).sensor_position == (51.5007, -0.1246)
    monkeypatch.setenv("SENSOR_LAT", "0")
    monkeypatch.setenv("SENSOR_LON", "0")
    assert Settings.from_env(tmp_path).sensor_position is None


# ---------------------------------------------------------------- C4.4 rate limits
def test_cot_limited_per_uid(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(tak_client.time, "monotonic", lambda: clock[0])
    s = tak_client.TakClient("127.0.0.1", _free_port(), "isr-hq")
    monkeypatch.setattr(s, "start", lambda: None)  # no network thread needed
    assert s.send("<a/>", uid="CHANGE-1") == "queued"
    assert s.send("<a/>", uid="CHANGE-1") == "rate_limited"
    assert s.send("<b/>", uid="CHANGE-2") == "queued"
    clock[0] += 4.9
    assert s.retry_in("CHANGE-1") > 0 and s.send("<a/>", uid="CHANGE-1") == "rate_limited"
    clock[0] += 0.2
    assert s.send("<a/>", uid="CHANGE-1") == "queued"
    assert s.status()["pending"] == 3


def test_mesh_one_per_30s(monkeypatch):
    clock = [500.0]
    monkeypatch.setattr(mesh.time, "monotonic", lambda: clock[0])
    sent = []
    from paho.mqtt import publish
    monkeypatch.setattr(publish, "single", lambda topic, payload, **kw: sent.append(json.loads(payload)))
    bus = mesh.MeshBus("127.0.0.1", 1883, "u", "p")
    assert bus.send("one") is True
    assert bus.send("two") is False and "rate limited" in bus.last_note and bus.last_error is None
    clock[0] += 29
    assert bus.send("three") is False
    clock[0] += 1.5
    assert bus.send("four") is True
    assert [p["text"] for p in sent] == ["one", "four"]


def test_mesh_failed_send_does_not_start_the_window():
    bus = mesh.MeshBus("127.0.0.1", _free_port(), "u", "p")
    assert bus.send("x") is False and bus.last_error and bus.retry_in() == 0


def test_sim_prefix():
    assert mesh.label_simulated("CHANGE ab: 3", True) == "[SIM] CHANGE ab: 3"
    assert mesh.label_simulated("[SIM] x", True) == "[SIM] x"
    assert mesh.label_simulated("CHANGE ab: 3", False) == "CHANGE ab: 3"


# ---------------------------------------------------------------- web: CD-1, C4.2, C4.4 end to end
def _app(tmp_path, sensor=None):
    st = Settings(data_dir=tmp_path, simulate=True, tak_port=_free_port(), mqtt_port=_free_port(),
                  yolo_model=tmp_path / "models" / "none.onnx", yolo_backend="opencv", vlm_backend="off",
                  sensor_position=sensor)
    app = create_app(st)
    svc = app.state.svc
    cots, texts = [], []
    real_send = svc.cot.send  # keep the real per-uid limiter; TAK is down, so events just queue
    svc.cot.send = lambda xml, uid=None: (cots.append(xml), real_send(xml, uid=uid))[1]
    svc.mesh._publish = lambda payload: (texts.append(payload["text"]), True)[1]
    return TestClient(app), cots, texts


def _remarks(xml: str) -> str:
    return xml.split("<remarks>")[1].split("</remarks>")[0]


def test_synthetic_sample_is_labelled_simulated(tmp_path):
    c, cots, texts = _app(tmp_path)
    res = c.post("/api/analyze", data={"sample": "synthetic"}).json()
    assert res["synthetic"] is True
    r = c.post(f"/api/results/{res['id']}/cot", json={})
    assert r.status_code == 200 and r.json()["simulated"] is True
    assert _remarks(cots[-1]).startswith("SIMULATED: ")
    m = c.post(f"/api/results/{res['id']}/mesh", json={})
    assert m.status_code == 200 and m.json()["sent"] and texts[-1].startswith("[SIM] ")


def test_real_upload_unlabelled_and_uses_sensor_position(tmp_path, flat_pair):
    c, cots, texts = _app(tmp_path, sensor=(51.5007, -0.1246))
    before, after, _ = flat_pair
    enc = lambda im: io.BytesIO(cv2.imencode(".png", im)[1].tobytes())
    res = c.post("/api/analyze", files={"before": ("b.png", enc(before), "image/png"),
                                        "after": ("a.png", enc(after), "image/png")}).json()
    assert res["synthetic"] is False and res["gps"] is None
    assert c.get("/api/status").json()["sensor"] == {"lat": 51.5007, "lon": -0.1246}
    r = c.post(f"/api/results/{res['id']}/cot", json={})
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["lat"], j["lon"]) == (51.5007, -0.1246) and "SENSOR_LAT" in j["position_source"]
    assert 'lat="51.5007000" lon="-0.1246000"' in cots[-1]
    assert not _remarks(cots[-1]).startswith("SIMULATED")
    # C4.4: the same uid again within 5 s is refused, clearly
    again = c.post(f"/api/results/{res['id']}/cot", json={})
    assert again.status_code == 429 and "wait" in again.json()["detail"]
    m = c.post(f"/api/results/{res['id']}/mesh", json={})
    assert m.status_code == 200 and not texts[-1].startswith("[SIM]") and "@51.50070,-0.12460" in texts[-1]
    m2 = c.post(f"/api/results/{res['id']}/mesh", json={})
    assert m2.status_code == 429 and "30 s" in m2.json()["detail"] and len(texts) == 1


def test_no_position_never_sends_zero(tmp_path, flat_pair):
    c, cots, _ = _app(tmp_path)
    before, after, _ = flat_pair
    enc = lambda im: io.BytesIO(cv2.imencode(".png", im)[1].tobytes())
    res = c.post("/api/analyze", files={"before": ("b.png", enc(before), "image/png"),
                                        "after": ("a.png", enc(after), "image/png")}).json()
    r = c.post(f"/api/results/{res['id']}/cot", json={})
    assert r.status_code == 400 and "SENSOR_LAT" in r.json()["detail"]
    assert c.post(f"/api/results/{res['id']}/cot", json={"lat": 0, "lon": 0}).status_code == 400
    assert cots == []


# ---------------------------------------------------------------- CD-2 offline models
def _wait_loaded(lab, timeout=5):
    t0 = time.time()
    while lab.error in ("loading",) and time.time() - t0 < timeout:
        time.sleep(0.02)


def test_missing_weights_offline_fail_fast_and_clear(monkeypatch, tmp_path):
    monkeypatch.setattr(labeler, "internet_reachable", lambda *a, **k: False)
    lab = labeler.UltralyticsLabeler(tmp_path, "yolo11m.pt", device="mps")
    _wait_loaded(lab)
    assert not lab.enabled
    assert "not downloaded" in lab.error and "--setup-only" in lab.error


def test_missing_m_weights_fall_back_to_nano(monkeypatch, tmp_path):
    loaded = []

    class FakeYOLO:
        def __init__(self, path):
            loaded.append(path)

        def to(self, device):
            return self

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    (tmp_path / "yolo11n.pt").write_bytes(b"x")
    lab = labeler.UltralyticsLabeler(tmp_path, "yolo11m.pt", device="mps")
    _wait_loaded(lab)
    assert lab.enabled and lab.weights == "yolo11n.pt" and loaded[0].endswith("yolo11n.pt")


def test_prefetch_wants_m_and_n_on_gpu(monkeypatch):
    monkeypatch.delenv("YOLO_WEIGHTS", raising=False)
    assert prefetch.wanted_pt_weights("mps") == ["yolo11m.pt", "yolo11n.pt"]
    assert prefetch.wanted_pt_weights("cpu") == ["yolo11n.pt"]


def test_prefetch_downloads_pt_weights(monkeypatch, tmp_path):
    from pathlib import Path

    def fake_download(dest):
        Path(dest).write_bytes(b"0" * 2_000_000)
        return str(dest)

    downloads = types.SimpleNamespace(attempt_download_asset=fake_download)
    ul = types.ModuleType("ultralytics")
    monkeypatch.setitem(sys.modules, "ultralytics", ul)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", types.SimpleNamespace(downloads=downloads))
    monkeypatch.setitem(sys.modules, "ultralytics.utils.downloads", downloads)
    monkeypatch.setattr(prefetch, "detect_accelerator", lambda: "mps")
    monkeypatch.setenv("YOLO_BACKEND", "auto")
    monkeypatch.delenv("YOLO_WEIGHTS", raising=False)
    assert prefetch.main(["--data-dir", str(tmp_path), "--strict"]) == 0
    assert (tmp_path / "models" / "yolo11m.pt").stat().st_size == 2_000_000
    assert (tmp_path / "models" / "yolo11n.pt").exists()


def test_prefetch_pi_onnx_copy_and_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO_BACKEND", "opencv")
    monkeypatch.delenv("YOLO_MODEL", raising=False)
    monkeypatch.delenv("YOLO_ONNX_URL", raising=False)
    data = tmp_path / "data"
    assert prefetch.main(["--data-dir", str(data)]) == 0          # missing is a warning...
    assert prefetch.main(["--data-dir", str(data), "--strict"]) == 1  # ...unless strict
    src = tmp_path / "repo-models"
    src.mkdir()
    (src / "yolo11n.onnx").write_bytes(b"onnx")
    assert prefetch.main(["--data-dir", str(data), "--from", str(src), "--strict"]) == 0
    assert (data / "models" / "yolo11n.onnx").read_bytes() == b"onnx"


def test_prefetch_off(monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO_BACKEND", "off")
    assert prefetch.main(["--data-dir", str(tmp_path), "--strict"]) == 0
