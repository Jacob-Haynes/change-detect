"""Contract v5: TakClient against a fake OpenTAKServer that enforces OTS 1.8.3's three rules:

1. CoT from a connection that is not a registered EUD is dropped (cot.sender_uid FK).
2. A connection is bound only if its first <contact endpoint=...> event arrives after the async
   channel opens (`channel_delay` after connect); an earlier one is ignored, so the connection stays unbound.
3. A registration whose callsign is already held by a different uid is silently refused.

It also applies OTS's marker-storage rules (C5.5): a relayed event is STORED only if its type
matches ^a-[fhupansjk]-[ZPAGSUF] or ^b-m-p (not b-m-p-s-p-loc), it has no <takv> and no
<contact endpoint>, its position is real (not 0,0), and every <link uid> is a registered EUD.
"""

import re
import socket
import threading
import time
import xml.etree.ElementTree as ET

import pytest

from changedetect import tak_client
from changedetect.tak_client import TakClient, build_registration_sa

EVENT_RE = re.compile(rb"<event\b.*?</event>", re.S)
STORABLE_TYPE = re.compile(r"^(a-[fhupansjk]-[ZPAGSUF]|b-m-p)")


class FakeOTS:
    def __init__(self, channel_delay=0.25):
        self.channel_delay = channel_delay
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        self.euds: dict[str, str] = {}  # uid -> callsign
        self.delivered: list[ET.Element] = []
        self.stored: list[ET.Element] = []
        self.dropped: list[str] = []
        self.registrations: list[tuple[int, float, ET.Element]] = []  # (conn no, seconds after connect, event)
        self.refused: list[str] = []
        self.early: list[str] = []
        self.connections = 0
        self._conns: list[socket.socket] = []
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            with self._lock:
                self.connections += 1
                n = self.connections
                self._conns.append(c)
            threading.Thread(target=self._serve, args=(c, n), daemon=True).start()

    def _serve(self, c, n):
        t0, buf, bound = time.monotonic(), b"", None
        while True:
            try:
                data = c.recv(65536)
            except OSError:
                return
            if not data:
                return
            buf += data
            while (m := EVENT_RE.search(buf)) is not None:
                buf = buf[m.end():]
                ev = ET.fromstring(m.group(0))
                contact = ev.find("detail/contact")
                with self._lock:
                    if contact is not None and contact.get("endpoint"):  # an EUD announcing itself
                        uid, cs = ev.get("uid"), contact.get("callsign")
                        if bound is None and time.monotonic() - t0 < self.channel_delay:
                            self.early.append(uid)  # rule 2: channel not open yet, not bound
                            continue
                        if self.euds.get(uid) != cs and cs in self.euds.values():
                            self.refused.append(uid)  # rule 3: callsign taken by another uid
                            continue
                        self.euds[uid] = cs
                        bound = uid
                        self.registrations.append((n, time.monotonic() - t0, ev))
                    elif bound is None:
                        self.dropped.append(ev.get("uid"))  # rule 1: not a registered EUD
                    else:
                        self.delivered.append(ev)
                        if self.storable(ev):
                            self.stored.append(ev)

    def storable(self, ev: ET.Element) -> bool:
        t = ev.get("type") or ""
        if not STORABLE_TYPE.match(t) or t.startswith("b-m-p-s-p-loc"):
            return False
        if ev.find("detail/takv") is not None:
            return False
        c = ev.find("detail/contact")
        if c is not None and c.get("endpoint"):
            return False
        p = ev.find("point")
        if p is None or (float(p.get("lat", 0)) == 0 and float(p.get("lon", 0)) == 0):
            return False
        return all(link.get("uid") in self.euds for link in ev.findall("detail/link"))

    def drop_all(self):
        with self._lock:
            for c in self._conns:
                try:
                    c.shutdown(socket.SHUT_RDWR)
                    c.close()
                except OSError:
                    pass
            self._conns.clear()

    def close(self):
        self._stop = True
        self.drop_all()
        self.srv.close()


@pytest.fixture
def ots():
    f = FakeOTS()
    yield f
    f.close()


def _until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def _marker(uid="CHANGE-1"):
    from changedetect import cot
    return cot.build_change_cot(uid, 51.4779, -0.0015, "CHANGE x (1)", "SIMULATED: Change detected: test")


def _raw(port):
    s = socket.create_connection(("127.0.0.1", port))
    return s


# ------------------------------------------------------------------ the fake really enforces the rules
def test_fake_ots_enforces_the_three_rules(ots):
    # one-shot sender, the old behaviour: connect, send, close -> dropped
    s = _raw(ots.port)
    s.sendall(_marker().encode())
    s.close()
    # registration too early -> ignored, and what follows is dropped
    s = _raw(ots.port)
    s.sendall(build_registration_sa("early", None, "t").encode() + _marker("CHANGE-2").encode())
    assert _until(lambda: "CHANGE-2" in ots.dropped)
    assert ots.early == ["isrkit-change-detect-early"]
    s.close()
    # someone else already holds callsign "CHANGE isr-hq" -> silently refused
    s = _raw(ots.port)
    time.sleep(0.3)
    taken = build_registration_sa("isr-hq", None, "t").replace('uid="isrkit-change-detect-isr-hq"', 'uid="other"')
    s.sendall(taken.encode())
    assert _until(lambda: ots.euds.get("other") == "CHANGE isr-hq")
    s2 = _raw(ots.port)
    time.sleep(0.3)
    s2.sendall(build_registration_sa("isr-hq", None, "t").encode() + _marker("CHANGE-3").encode())
    assert _until(lambda: "CHANGE-3" in ots.dropped)
    assert ots.refused == ["isrkit-change-detect-isr-hq"] and ots.delivered == []
    s.close()
    s2.close()


# ------------------------------------------------------------------ C5.1-C5.3
def test_registers_after_delay_then_delivers_on_one_connection(ots):
    c = TakClient("127.0.0.1", ots.port, "isr-hq", (51.4779, -0.0015), "1.2.3", register_delay=0.3)
    c.start()
    try:
        assert _until(lambda: c.status()["connected"])  # C5.1: opened at startup, before any push
        assert c.send(_marker("CHANGE-1"), uid="CHANGE-1") == "queued"
        time.sleep(0.1)
        assert ots.delivered == [] and not c.status()["registered"]  # held back until registered
        assert _until(lambda: len(ots.delivered) == 1)
        assert c.status()["registered"] and c.status()["pending"] == 0
        n, after, sa = ots.registrations[0]
        assert after >= ots.channel_delay and sa.get("uid") == "isrkit-change-detect-isr-hq"
        assert ots.delivered[0].get("uid") == "CHANGE-1"
        assert "SIMULATED: " in ots.delivered[0].find("detail/remarks").text  # labels kept
        # C5.3: later markers use the same connection, and the per-uid rate limit still applies
        assert c.send(_marker("CHANGE-1"), uid="CHANGE-1") == "rate_limited"
        assert c.send(_marker("CHANGE-2"), uid="CHANGE-2") == "queued"
        assert _until(lambda: len(ots.delivered) == 2)
        assert ots.connections == 1 and ots.dropped == [] and ots.early == [] and ots.refused == []
    finally:
        c.stop()


def test_sa_resent_and_reregisters_after_reconnect(ots):
    c = TakClient("127.0.0.1", ots.port, "isr-hq", None, "1", register_delay=0.3, sa_interval=0.3)
    c.start()
    try:
        assert _until(lambda: len(ots.registrations) >= 3)  # periodic SA on one connection
        assert {r[0] for r in ots.registrations} == {1}
        ots.drop_all()
        assert _until(lambda: ots.connections >= 2 and any(r[0] >= 2 for r in ots.registrations), timeout=6)
        reg = [r for r in ots.registrations if r[0] >= 2][0]
        assert reg[1] >= ots.channel_delay  # waited again after reconnect
        c.send(_marker("CHANGE-9"), uid="CHANGE-9")
        assert _until(lambda: any(e.get("uid") == "CHANGE-9" for e in ots.delivered))
        assert ots.refused == [] and ots.early == []  # same stable uid, so the callsign is ours again
    finally:
        c.stop()


def test_restart_reuses_the_stable_identity(ots):
    for _ in range(2):
        c = TakClient("127.0.0.1", ots.port, "isr-hq", None, "1", register_delay=0.3)
        c.start()
        before = len(ots.registrations)
        assert _until(lambda: len(ots.registrations) > before)
        c.stop()
    assert ots.refused == [] and list(ots.euds) == ["isrkit-change-detect-isr-hq"]


def test_markers_queue_while_tak_is_down():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    c = TakClient("127.0.0.1", port, "isr-hq", register_delay=0.3, max_backoff=0.5)
    c.start()
    try:
        c.send(_marker(), uid="CHANGE-1")
        time.sleep(0.4)
        st = c.status()
        assert not st["connected"] and st["pending"] == 1 and st["last_error"]
        ots = FakeOTS()
        try:  # the server comes up on another port in this test, so point the client at it
            c.port = ots.port
            assert _until(lambda: len(ots.delivered) == 1, timeout=6)
        finally:
            ots.close()
    finally:
        c.stop()


# ------------------------------------------------------------------ C5.2 registration SA format
def _times(ev):
    return [ev.get(k) for k in ("time", "start", "stale")]


def test_registration_sa_with_position():
    xml = build_registration_sa("isr-hq", (51.4779, -0.0015), "0.1.0", now=1_800_000_000.0)
    assert xml == build_registration_sa("isr-hq", (51.4779, -0.0015), "0.1.0", now=1_800_000_000.0)
    ev = ET.fromstring(xml.split("?>", 1)[1])
    assert ev.get("uid") == "isrkit-change-detect-isr-hq" and ev.get("type") == "a-f-G-E-S"
    c = ev.find("detail/contact")
    assert c.get("callsign") == "CHANGE isr-hq" and c.get("endpoint") == "*:-1:stcp"
    g = ev.find("detail/__group")
    assert (g.get("name"), g.get("role")) == ("Cyan", "Team Member")
    t = ev.find("detail/takv")
    assert (t.get("platform"), t.get("device"), t.get("version")) == ("isr-kit", "change-detect", "0.1.0") and t.get("os")
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", x) for x in _times(ev))
    p = ev.find("point")
    assert float(p.get("lat")) == 51.4779 and float(p.get("lon")) == -0.0015
    assert ev.get("start") == tak_client.iso(1_800_000_000.0)
    assert ev.get("stale") == tak_client.iso(1_800_000_000.0 + 120)


def test_registration_sa_without_position_is_never_drawn():
    ev = ET.fromstring(build_registration_sa("isr-pi", None, "0.1.0").split("?>", 1)[1])
    p = ev.find("point")
    assert (p.get("lat"), p.get("lon"), p.get("hae"), p.get("ce"), p.get("le")) == \
        ("0", "0", "9999999", "9999999", "9999999")
    assert ev.get("stale") == ev.get("start")
    assert ev.find("detail/contact").get("callsign") == "CHANGE isr-pi"


# ------------------------------------------------------------------ end to end through the web app
def test_web_app_connects_at_startup_and_marker_reaches_ots(ots, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from changedetect.config import Settings
    from changedetect.web.app import create_app
    monkeypatch.setenv("ISR_HOST", "isr-hq")
    st = Settings(data_dir=tmp_path, simulate=True, tak_port=ots.port, mqtt_port=1, yolo_backend="off",
                  vlm_backend="off", heartbeat_enabled=False)
    app = create_app(st)
    svc = app.state.svc
    try:
        assert _until(lambda: ots.connections == 1)  # opened at startup, nothing pushed yet
        web = TestClient(app)
        rid = web.post("/api/analyze", data={"sample": "synthetic"}).json()["id"]
        assert web.post(f"/api/results/{rid}/cot", json={}).json()["state"] == "queued"
        assert _until(lambda: len(ots.delivered) == 1, timeout=8)  # after the real 3 s registration delay
        reg = ots.registrations[0]
        assert reg[1] >= 3.0 and reg[2].find("detail/contact").get("callsign") == "CHANGE isr-hq"
        ev = ots.delivered[0]
        assert ev.get("uid") == f"CHANGE-{rid}" and ev.find("detail/remarks").text.startswith("SIMULATED: ")
        assert web.get("/api/status").json()["tak"]["registered"] is True
    finally:
        svc.cot.stop()


# ------------------------------------------------------------------ C5.5 markers OTS stores
def _ev(xml):
    return ET.fromstring(xml.split("?>", 1)[-1])


def test_fake_ots_storage_rules(ots):
    from changedetect import cot
    ots.euds["isrkit-change-detect-isr-hq"] = "CHANGE isr-hq"
    good = cot.build_change_cot("CHANGE-1", 51.4779, -0.0015, "CHANGE 1", "r",
                                parent_uid="isrkit-change-detect-isr-hq", parent_callsign="CHANGE isr-hq")
    assert ots.storable(_ev(good))
    assert not ots.storable(_ev(build_registration_sa("isr-hq", (51.4, -0.1), "1")))  # takv + endpoint
    assert not ots.storable(_ev(good.replace('type="a-u-G"', 'type="b-m-p-s-p-loc"')))
    assert not ots.storable(_ev(good.replace('type="a-u-G"', 'type="t-x-c"')))
    assert not ots.storable(_ev(good.replace('uid="isrkit-change-detect-isr-hq"', 'uid="CHANGE-0"')))
    assert not ots.storable(_ev(good.replace('lat="51.4779000" lon="-0.0015000"', 'lat="0" lon="0"')))
    with pytest.raises(ValueError):
        cot.build_change_cot("CHANGE-2", 0, 0, "x", "r")


def test_change_marker_from_web_app_is_stored(ots, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from changedetect.config import Settings
    from changedetect.web.app import create_app
    monkeypatch.setenv("ISR_HOST", "isr-hq")
    st = Settings(data_dir=tmp_path, simulate=True, tak_port=ots.port, mqtt_port=1, yolo_backend="off",
                  vlm_backend="off", heartbeat_enabled=False)
    app = create_app(st)
    try:
        web = TestClient(app)
        rid = web.post("/api/analyze", data={"sample": "synthetic"}).json()["id"]
        assert web.post(f"/api/results/{rid}/cot", json={}).status_code == 200
        manual = web.post("/api/analyze", data={"sample": "synthetic"}).json()["id"]
        assert web.post(f"/api/results/{manual}/cot", json={"lat": 51.5007, "lon": -0.1246}).status_code == 200
        assert _until(lambda: len(ots.delivered) == 2, timeout=8)
        assert len(ots.stored) == 2, [ET.tostring(e) for e in ots.delivered]
        for ev in ots.stored:
            assert ev.get("type") == "a-u-G" and ev.find("detail/takv") is None
            assert ev.find("detail/contact").get("endpoint") is None
            (link,) = ev.findall("detail/link")
            assert link.get("uid") == "isrkit-change-detect-isr-hq" and link.get("relation") == "p-p"
    finally:
        app.state.svc.cot.stop()
