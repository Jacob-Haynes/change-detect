"""CoT XML over TCP to the TAK server, with a background sender and reconnect backoff.

If the TAK server is down, markers wait in a small queue and are retried; nothing blocks
or crashes the web app.
"""

from __future__ import annotations

import collections
import logging
import socket
import threading
import time
from datetime import datetime, timezone
from xml.sax.saxutils import escape, quoteattr

logger = logging.getLogger(__name__)

COT_TYPE = "a-u-G"  # unknown ground point: a change, not an identification
STALE_SECONDS = 6 * 3600
MAX_PENDING = 50
MIN_INTERVAL_PER_UID_S = 5.0  # contract C4.4: at most one event per uid every 5 s


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def build_change_cot(uid: str, lat: float, lon: float, callsign: str, remarks: str,
                     manual: bool = False, ce: float = 25.0, now: float | None = None) -> str:
    now = time.time() if now is None else now
    how = "h-e" if manual else "m-g"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(uid)} type="{COT_TYPE}" how="{how}" '
        f'time="{_iso(now)}" start="{_iso(now)}" stale="{_iso(now + STALE_SECONDS)}">'
        f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="9999999.0" ce="{ce:.1f}" le="9999999.0"/>'
        f'<detail><contact callsign={quoteattr(callsign)}/>'
        f'<remarks>{escape(remarks)}</remarks>'
        f'<__group name="Yellow" role="Team Member"/></detail></event>'
    )


class CotSender:
    def __init__(self, host: str, port: int, enabled: bool = True):
        self.host, self.port, self.enabled = host, port, enabled
        self._q: collections.deque[str] = collections.deque(maxlen=MAX_PENDING)
        self._cv = threading.Condition()
        self._sock: socket.socket | None = None
        self.connected = False
        self.last_error: str | None = None
        self.sent = 0
        self._backoff = 1.0
        self._stop = False
        self._thread: threading.Thread | None = None
        self._last_by_uid: dict[str, float] = {}

    def start(self) -> None:
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="cot-sender", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        self._close()

    def retry_in(self, uid: str, now: float | None = None) -> float:
        """Seconds until another event for this uid is allowed (0 = now)."""
        last = self._last_by_uid.get(uid)
        if last is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, MIN_INTERVAL_PER_UID_S - (now - last))

    def send(self, xml: str, uid: str | None = None) -> str:
        """Queue a CoT event. Returns 'queued', 'disabled' or 'rate_limited' (same uid within 5 s)."""
        if not self.enabled:
            return "disabled"
        with self._cv:
            if uid is not None:
                if self.retry_in(uid) > 0:
                    return "rate_limited"
                self._last_by_uid[uid] = time.monotonic()
            self._q.append(xml)
            self._cv.notify_all()
        self.start()
        return "queued"

    def status(self) -> dict:
        return {"enabled": self.enabled, "host": self.host, "port": self.port, "connected": self.connected,
                "pending": len(self._q), "sent": self.sent, "last_error": self.last_error}

    def _connect(self) -> bool:
        try:
            s = socket.create_connection((self.host, self.port), timeout=5)
            s.settimeout(10)
            self._sock = s
            self.connected = True
            self.last_error = None
            self._backoff = 1.0
            logger.info("Connected to TAK server %s:%d", self.host, self.port)
            return True
        except OSError as e:
            self.last_error = f"TAK server unavailable: {e}"
            self.connected = False
            return False

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self.connected = False

    def _run(self) -> None:
        while not self._stop:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait(timeout=30)
                if self._stop:
                    return
                xml = self._q[0]
            if self._sock is None and not self._connect():
                logger.warning("%s (retry in %.0fs, %d pending)", self.last_error, self._backoff, len(self._q))
                time.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, 60.0)
                continue
            try:
                self._sock.sendall(xml.encode("utf-8"))
                self.sent += 1
                with self._cv:
                    if self._q and self._q[0] is xml:
                        self._q.popleft()
            except OSError as e:
                self.last_error = f"TAK send failed: {e}"
                logger.warning(self.last_error)
                self._close()
