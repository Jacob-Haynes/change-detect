import io
import socket

import cv2
import pytest
from fastapi.testclient import TestClient

from changedetect.config import Settings
from changedetect.web.app import create_app


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    st = Settings(data_dir=d, simulate=True, tak_port=_free_port(), mqtt_port=_free_port(),
                  yolo_model=d / "models" / "none.onnx")
    return TestClient(create_app(st))


def test_index_and_status(client):
    assert "CHANGE DETECT" in client.get("/").text
    s = client.get("/api/status").json()
    assert s["simulate"] and not s["gemini"]["enabled"] and not s["yolo"]["enabled"]


def test_sample_analysis_and_actions(client):
    r = client.post("/api/analyze", data={"sample": "synthetic"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["changes_total"] >= 4
    rid = res["id"]
    v = res["views"][0]
    assert client.get(f"/results/{rid}/{v['images']['heat']}").status_code == 200
    assert client.get(f"/results/{rid}/report.pdf").status_code == 200
    assert client.get("/api/results").json()[0]["id"] == rid
    # Gemini off -> clear 400
    assert client.post(f"/api/results/{rid}/ai").status_code == 400
    # ATAK: GPS from EXIF, TAK down -> queued, no crash
    c = client.post(f"/api/results/{rid}/cot", json={})
    assert c.status_code == 200 and c.json()["state"] == "queued"
    assert client.post(f"/api/results/{rid}/cot", json={"lat": 95, "lon": 0}).status_code == 400
    # Mesh: broker down -> sent False, no crash
    m = client.post(f"/api/results/{rid}/mesh", json={})
    assert m.status_code == 200 and m.json()["sent"] is False and len(m.json()["text"]) <= 200


def test_upload_without_gps_needs_manual_position(client, flat_pair):
    before, after, _ = flat_pair
    enc = lambda im: io.BytesIO(cv2.imencode(".png", im)[1].tobytes())
    r = client.post("/api/analyze", files={"before": ("b.png", enc(before), "image/png"),
                                           "after": ("a.png", enc(after), "image/png")},
                    data={"use_yolo": "false"})
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    assert r.json()["gps"] is None
    assert client.post(f"/api/results/{rid}/cot", json={}).status_code == 400
    ok = client.post(f"/api/results/{rid}/cot", json={"lat": 51.1, "lon": -1.7})
    assert ok.status_code == 200 and ok.json()["lat"] == 51.1


def test_bad_inputs(client):
    assert client.post("/api/analyze", data={}).status_code == 400
    r = client.post("/api/analyze", files={"before": ("x.txt", io.BytesIO(b"hi")), "after": ("y.txt", io.BytesIO(b"hi"))})
    assert r.status_code == 400
    assert client.get("/api/results/../../etc").status_code == 404
    assert client.get("/api/results/nope-nope").status_code == 404
