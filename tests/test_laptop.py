import http.server
import json
import socket
import sys
import threading
import types

import cv2
import numpy as np
import pytest

from changedetect import align, batch, heartbeat, labeler, media, pipeline, synthetic, vlm
from changedetect.config import Settings
from changedetect.gemini import GeminiAdvisor


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------- profiles / resolution

def test_laptop_profile_keeps_full_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("CHANGE_DETECT_PROFILE", "laptop")
    st = Settings.from_env(tmp_path)
    assert st.profile == "laptop" and st.limits["max_dim"] is None
    monkeypatch.delenv("CHANGE_DETECT_PROFILE")
    assert Settings.from_env(tmp_path).limits["max_dim"] == 1600


def test_full_res_alignment_rescales_homography(flat_pair, tmp_path):
    before, after, truth = flat_pair
    big_b = cv2.resize(before, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)  # 3600 px
    big_a = cv2.resize(after, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    al = align.align(big_b, big_a)
    assert al.ok and al.grade == "good"
    small = align.align(before, after)
    # same transform, expressed at 2.5x: translation scales, the linear part does not
    assert np.allclose(np.array(al.H)[:2, :2], np.array(small.H)[:2, :2], atol=0.02)
    cv2.imwrite(str(tmp_path / "b.png"), big_b)
    cv2.imwrite(str(tmp_path / "a.png"), big_a)
    r = pipeline.analyze(tmp_path / "b.png", tmp_path / "a.png", tmp_path / "out",
                         pipeline.Options(max_dim=None, max_dim_360=None))
    assert r["views"][0]["image_size"][0] == 3600
    assert r["changes_total"] >= 4


def test_heic_gps(tmp_path):
    from PIL import Image
    im = Image.fromarray((np.random.default_rng(0).random((200, 300, 3)) * 255).astype("uint8"))
    ex = Image.Exif()
    g = ex.get_ifd(0x8825)
    g[1], g[2], g[3], g[4] = "N", (51.0, 12.0, 0.0), "W", (1.0, 48.0, 0.0)
    im.save(tmp_path / "IMG_0001.HEIC", exif=ex)
    m = media.read_exif(tmp_path / "IMG_0001.HEIC")
    assert m.lat == pytest.approx(51.2) and m.lon == pytest.approx(-1.8)
    assert media.load_image(tmp_path / "IMG_0001.HEIC").shape == (200, 300, 3)


def test_insv_gets_helpful_error():
    with pytest.raises(media.MediaError, match="equirectangular"):
        media.kind_of("VID_001.insv")


# ---------------------------------------------------------------- batches

def _item(name, ts, path=None):
    return batch.Item(path=path, name=name, ts=ts)


def test_split_by_time_gap():
    items = [_item(f"G{i}", 1000 + i * 20) for i in range(5)] + [_item(f"H{i}", 9000 + i * 25) for i in range(4)]
    b, a, info = batch.split_by_time(items)
    assert [i.name for i in b] == [f"G{i}" for i in range(5)] and len(a) == 4
    assert info["split_gap_s"] > 7000


def test_split_refuses_without_clear_gap():
    with pytest.raises(ValueError):
        batch.split_by_time([_item(f"G{i}", 1000 + i * 30) for i in range(6)])
    with pytest.raises(ValueError):
        batch.split_by_time([_item("a", None), _item("b", None)])


def test_pairing_by_content(tmp_path):
    paths_b, paths_a = [], []
    for k, seed in enumerate((3, 5, 9)):
        b, a, _ = synthetic.make_pair(seed=seed)
        pb, pa = tmp_path / f"b{k}.jpg", tmp_path / f"a{k}.jpg"
        cv2.imwrite(str(pb), b)
        cv2.imwrite(str(pa), a)
        paths_b.append(pb)
        paths_a.append(pa)
    before = [batch.Item(path=p, name=p.name, ts=i) for i, p in enumerate(paths_b)]
    after = [batch.Item(path=p, name=p.name, ts=100 + i) for i, p in enumerate(reversed(paths_a))]
    pairs, ub, ua = batch.pair_items(before, after)
    assert {(p["before"], p["after"]) for p in pairs} == {("b0.jpg", "a0.jpg"), ("b1.jpg", "a1.jpg"), ("b2.jpg", "a2.jpg")}
    assert not ub and not ua


# ---------------------------------------------------------------- accelerator / YOLO backend

@pytest.mark.parametrize("cuda,mps,expect", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_accelerator_order(monkeypatch, cuda, mps, expect):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda),
                                 backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert labeler.detect_accelerator() == expect
    assert labeler.default_weights(expect) == ("yolo11n.pt" if expect == "cpu" else "yolo11m.pt")


def test_make_labeler_falls_back_to_onnx(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # import fails
    lab = labeler.make_labeler("auto", tmp_path / "missing.onnx", tmp_path)
    assert isinstance(lab, labeler.Labeler) and not lab.enabled
    assert not labeler.make_labeler("off", None, tmp_path).enabled


# ---------------------------------------------------------------- VLM chain

class _FakeOllama(http.server.BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "qwen3-vl:8b"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.seen.append(req)
        self._send({"choices": [{"message": {"content": "<think>hmm</think>\nVehicle has left the track."}}]})


def test_vlm_openai_compat_then_off():
    port = _free_port()
    srv = http.server.HTTPServer(("127.0.0.1", port), _FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        chain = vlm.AdvisorChain("auto", vlm.OpenAICompatAdvisor(f"http://127.0.0.1:{port}/v1", "qwen3-vl:8b"),
                                 GeminiAdvisor(None, "gemini-3.8-flash"))
        assert chain.active() == "openai_compat"
        text, source = chain.describe(b"\xff\xd8before", b"\xff\xd8after", "new object")
        assert text == "Vehicle has left the track." and "qwen3-vl" in source
        content = _FakeOllama.seen[-1]["messages"][0]["content"]
        assert sum(1 for c in content if c["type"] == "image_url") == 2
    finally:
        srv.shutdown()
    dead = vlm.AdvisorChain("auto", vlm.OpenAICompatAdvisor(f"http://127.0.0.1:{_free_port()}/v1", "x"),
                            GeminiAdvisor(None, "m"))
    assert dead.active() == "off" and dead.describe(b"", b"") == (None, None)


def test_vlm_falls_back_to_gemini_when_local_down():
    chain = vlm.AdvisorChain("auto", vlm.OpenAICompatAdvisor(f"http://127.0.0.1:{_free_port()}/v1", "x"),
                             GeminiAdvisor("fake-key", "gemini-3.8-flash"))
    assert chain.active() == "gemini" and chain.status()["cloud"]
    assert vlm.AdvisorChain("off", chain.local, chain.gemini).active() == "off"


# ---------------------------------------------------------------- heartbeat

def test_heartbeat_payload_contract():
    p = heartbeat.build_payload("hq-mac", "http://10.42.0.20:8130/", "mps", "ok", ts=123)
    assert p == {"tool": "change-detect", "host": "hq-mac", "url": "http://10.42.0.20:8130/",
                 "mode": "mps", "status": "ok", "ts": 123}
    assert heartbeat.TOPIC == "isrkit/heartbeat/change-detect"


def test_heartbeat_broker_down_is_harmless():
    hb = heartbeat.Heartbeat("127.0.0.1", _free_port(), "isrkit", "x", 8130, lambda: "cpu", lambda: "ok")
    hb.start()
    import time
    time.sleep(0.5)
    assert not hb.status()["connected"]
    assert hb.payload()["url"].startswith("http://") and hb.payload()["url"].endswith(":8130/")
    hb.stop()
