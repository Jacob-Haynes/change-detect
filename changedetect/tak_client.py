"""Long-lived CoT client for OpenTAKServer (contract v5). Standard library only.

OpenTAKServer drops CoT from senders that are not a registered EUD, binds a client's receive queue
only if its first <contact> event arrives after the server-side channel is up (about 3 s after
connect), and silently refuses a uid whose callsign is already taken. So this client:

- C5.1 keeps ONE TCP connection to TAK_HOST:TAK_PORT open from startup, reconnecting with backoff;
- C5.2 on every (re)connect waits `register_delay` (3 s), then sends a registration SA with a stable
  uid `isrkit-change-detect-<host>` and callsign `CHANGE <host>`, and re-sends it every 60 s.
  Position = SENSOR_LAT/LON with stale now+120 s; if unset, the "no position" point
  (0,0 with hae/ce/le 9999999 and stale = start, so ATAK never draws it);
- C5.3 sends change markers on the same connection, only once registered. Markers queue while
  disconnected or registering. At most one marker per uid every 5 s (contract C4.4).

Anything the server sends is read and discarded, so its buffers never fill.
"""

from __future__ import annotations

import collections
import logging
import platform
import select
import socket
import threading
import time
from datetime import datetime, timezone
from xml.sax.saxutils import quoteattr

logger = logging.getLogger(__name__)

SA_TYPE = "a-f-G-E-S"
REGISTER_DELAY_S = 3.0
SA_INTERVAL_S = 60.0
SA_STALE_S = 120.0
MAX_PENDING = 50
MIN_INTERVAL_PER_UID_S = 5.0
NO_POS = 9999999


def iso(ts: float) -> str:
    """ISO 8601 UTC with milliseconds, ending in Z."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def registration_uid(host: str) -> str:
    return f"isrkit-change-detect-{host}"


def registration_callsign(host: str) -> str:
    return f"CHANGE {host}"


def build_registration_sa(host: str, position: tuple[float, float] | None, version: str,
                          now: float | None = None) -> str:
    now = time.time() if now is None else now
    if position:
        lat, lon = position
        point = f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="{NO_POS}" ce="25" le="{NO_POS}"/>'
        stale = now + SA_STALE_S
    else:  # the only 0,0 allowed in the kit: flagged unknown and already stale, so never drawn
        point = f'<point lat="0" lon="0" hae="{NO_POS}" ce="{NO_POS}" le="{NO_POS}"/>'
        stale = now
    os_name = f"{platform.system()} {platform.release()}".strip()
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(registration_uid(host))} type="{SA_TYPE}" how="m-g" '
        f'time="{iso(now)}" start="{iso(now)}" stale="{iso(stale)}">'
        f'{point}'
        f'<detail><contact callsign={quoteattr(registration_callsign(host))} endpoint="*:-1:stcp"/>'
        f'<__group name="Cyan" role="Team Member"/>'
        f'<takv platform="isr-kit" device="change-detect" os={quoteattr(os_name)} version={quoteattr(version)}/>'
        f'</detail></event>'
    )


class TakClient:
    def __init__(self, host: str, port: int, station: str, position: tuple[float, float] | None = None,
                 version: str = "0", enabled: bool = True, register_delay: float = REGISTER_DELAY_S,
                 sa_interval: float = SA_INTERVAL_S, max_backoff: float = 60.0):
        self.host, self.port, self.station, self.position = host, port, station, position
        self.version, self.enabled = version, enabled
        self.register_delay, self.sa_interval, self.max_backoff = register_delay, sa_interval, max_backoff
        self.uid, self.callsign = registration_uid(station), registration_callsign(station)
        self.connected = False
        self.registered = False
        self.last_error: str | None = None
        self.sent = 0
        self.connects = 0
        self._q: collections.deque[str] = collections.deque(maxlen=MAX_PENDING)
        self._last_by_uid: dict[str, float] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ public API
    def start(self) -> None:
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="tak-client", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        self._close()
        if self._thread is not None:
            self._thread.join(timeout)

    def retry_in(self, uid: str, now: float | None = None) -> float:
        last = self._last_by_uid.get(uid)
        if last is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, MIN_INTERVAL_PER_UID_S - (now - last))

    def send(self, xml: str, uid: str | None = None) -> str:
        """Queue a marker. Returns 'queued', 'disabled' or 'rate_limited' (same uid within 5 s)."""
        if not self.enabled:
            return "disabled"
        with self._lock:
            if uid is not None:
                if self.retry_in(uid) > 0:
                    return "rate_limited"
                self._last_by_uid[uid] = time.monotonic()
            self._q.append(xml)
        self._wake.set()
        self.start()
        return "queued"

    def status(self) -> dict:
        return {"enabled": self.enabled, "host": self.host, "port": self.port, "connected": self.connected,
                "registered": self.registered, "uid": self.uid, "callsign": self.callsign,
                "pending": len(self._q), "sent": self.sent, "last_error": self.last_error}

    # ------------------------------------------------------------------ internals
    def _close(self) -> None:
        s, self._sock = self._sock, None
        self.connected = self.registered = False
        if s is not None:
            try:
                s.close()
            except OSError:
                pass

    @staticmethod
    def _write(sock: socket.socket, data: str) -> None:
        sock.sendall(data.encode("utf-8"))

    def _drain(self, sock: socket.socket, timeout: float) -> None:
        """Wait up to `timeout` for incoming data or a wake-up; discard what the server sends."""
        end = time.monotonic() + timeout
        while not self._stop.is_set():
            left = end - time.monotonic()
            if left <= 0:
                return
            r, _, _ = select.select([sock], [], [], min(left, 0.1))
            if r:
                if not sock.recv(65536):
                    raise ConnectionError("TAK server closed the connection")
            if self._wake.is_set():
                return

    def _register(self, sock: socket.socket) -> None:
        self._write(sock, build_registration_sa(self.station, self.position, self.version))
        self.registered = True

    def _session(self) -> None:
        # the worker only uses this local reference; stop() may close self._sock from another thread
        sock = self._sock = socket.create_connection((self.host, self.port), timeout=5)
        sock.settimeout(10)
        self.connected, self.connects = True, self.connects + 1
        self.last_error = None
        logger.info("Connected to TAK server %s:%d; registering as %s in %.0f s",
                    self.host, self.port, self.callsign, self.register_delay)
        # C5.2: let the server open its channel before the first <contact>
        end = time.monotonic() + self.register_delay
        while not self._stop.is_set() and time.monotonic() < end:
            self._wake.clear()
            self._drain(sock, end - time.monotonic())
        if self._stop.is_set():
            return
        self._register(sock)
        next_sa = time.monotonic() + self.sa_interval
        while not self._stop.is_set():
            while True:  # C5.3: markers only after registration, on this connection
                with self._lock:
                    xml = self._q[0] if self._q else None
                if xml is None:
                    break
                self._write(sock, xml)
                self.sent += 1
                with self._lock:
                    if self._q and self._q[0] is xml:
                        self._q.popleft()
            if time.monotonic() >= next_sa:
                self._register(sock)
                next_sa = time.monotonic() + self.sa_interval
            self._wake.clear()
            if self._q:
                continue
            self._drain(sock, max(0.0, next_sa - time.monotonic()))

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._session()
                backoff = 1.0
            except (OSError, ValueError) as e:  # ConnectionError, timeouts, or a socket closed by stop()
                if not self._stop.is_set():
                    self.last_error = f"TAK server unavailable: {e}"
                    logger.warning("%s (retry in %.0f s, %d pending)", self.last_error, backoff, len(self._q))
                if self.connected:
                    backoff = 1.0  # we had a session: reconnect promptly
            self._close()
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self.max_backoff)
