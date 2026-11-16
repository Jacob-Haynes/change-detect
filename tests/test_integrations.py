import json
import socket
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from changedetect import cot, gemini, labeler, mesh


def test_cot_xml_is_valid():
    xml = cot.build_change_cot("change-detect-x", 51.48, -0.01, "CHANGE <1>", 'car "gone" & shelter', manual=True)
    ev = ET.fromstring(xml.split("?>", 1)[1])
    assert ev.get("type") == "a-u-G" and ev.get("how") == "h-e"
    assert float(ev.find("point").get("lat")) == 51.48
    assert ev.find("detail/contact").get("callsign") == "CHANGE <1>"
    assert "&" in ev.find("detail/remarks").text


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_cot_sender_survives_tak_down_then_delivers():
    port = _free_port()
    sender = cot.CotSender("127.0.0.1", port)
    assert sender.send("<event/>") == "queued"
    time.sleep(0.3)
    assert not sender.status()["connected"] and sender.status()["pending"] == 1
    got = []
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)

    def accept():
        c, _ = srv.accept()
        got.append(c.recv(100))
        c.close()

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    t.join(timeout=6)
    sender.stop()
    srv.close()
    assert got and got[0] == b"<event/>"


def test_cot_disabled():
    assert cot.CotSender("127.0.0.1", 1, enabled=False).send("<event/>") == "disabled"


def test_mesh_payload_contract():
    p = mesh.build_payload("x " * 300, to=None)
    assert set(p) == {"text", "to", "channel", "source"}
    assert len(p["text"]) <= 200 and p["source"] == "change-detect" and p["channel"] == 0
    json.dumps(p)


def test_mesh_broker_down_does_not_raise():
    bus = mesh.MeshBus("127.0.0.1", _free_port(), "guest", "guest")
    assert bus.send("hello") is False
    assert bus.last_error


def test_gemini_off_without_key():
    g = gemini.GeminiAdvisor(None, "gemini-3.8-flash")
    assert not g.enabled
    assert g.describe(b"", b"") is None
    assert g.status()["model"] is None


def test_yolo_decode_and_describe():
    out = np.zeros((1, 84, 3), np.float32)
    out[0, :4, 0] = [320, 320, 64, 32]   # car at centre of a 640 input
    out[0, 4 + 2, 0] = 0.9
    out[0, :4, 1] = [100, 100, 20, 40]   # weak person: filtered
    out[0, 4 + 0, 1] = 0.1
    dets = labeler.decode_yolo(out, 1280, 960)
    assert len(dets) == 1 and dets[0].group == "vehicle"
    x, y, w, h = dets[0].box
    assert abs(x - 576) <= 1 and abs(w - 128) <= 1
    labels, text = labeler.describe((570, 450, 140, 60), [], dets)
    assert text == "vehicle appeared" and labels
    _, text = labeler.describe((570, 450, 140, 60), dets, [])
    assert text == "vehicle gone"


def test_labeler_missing_model_is_disabled(tmp_path):
    lab = labeler.Labeler(tmp_path / "nope.onnx")
    assert not lab.enabled and lab.detect(np.zeros((10, 10, 3), np.uint8)) == []
