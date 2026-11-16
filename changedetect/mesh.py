"""Publish short mesh texts via the meshbus MQTT topic (the blufor-tracker bridge owns the radio)."""

from __future__ import annotations

import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

TX_TOPIC = "meshbus/tx"
MAX_LEN = 200
MIN_INTERVAL_S = 30.0  # contract C4.4: at most one mesh text per tool every 30 s
SIM_PREFIX = "[SIM] "  # contract C4.3: texts about synthetic/simulated data


def label_simulated(text: str, simulated: bool) -> str:
    if simulated and not text.startswith(SIM_PREFIX):
        return SIM_PREFIX + text
    return text


def build_payload(text: str, to: str | None = None, channel: int = 0) -> dict:
    text = " ".join(text.split())
    if len(text) > MAX_LEN:
        text = text[: MAX_LEN - 1] + "…"
    return {"text": text, "to": to, "channel": channel, "source": "change-detect"}


class MeshBus:
    def __init__(self, host: str, port: int, user: str, password: str, enabled: bool = True):
        self.host, self.port, self.user, self.password, self.enabled = host, port, user, password, enabled
        self.last_error: str | None = None
        self.sent = 0
        self.last_note: str | None = None
        self._last_sent: float | None = None
        self._lock = threading.Lock()

    def retry_in(self, now: float | None = None) -> float:
        """Seconds until the next mesh text is allowed (0 = now)."""
        if self._last_sent is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, MIN_INTERVAL_S - (now - self._last_sent))

    def send(self, text: str, to: str | None = None, channel: int = 0) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            wait = self.retry_in()
            if wait > 0:
                self.last_note = f"rate limited: one mesh text per {MIN_INTERVAL_S:.0f} s, next in {wait:.0f} s"
                return False
            self.last_note = None
            ok = self._publish(build_payload(text, to, channel))
            if ok:
                self.sent += 1
                self._last_sent = time.monotonic()
            return ok

    def _publish(self, payload: dict) -> bool:
        try:
            from paho.mqtt import publish
            publish.single(TX_TOPIC, json.dumps(payload), qos=1, hostname=self.host, port=self.port,
                           auth={"username": self.user, "password": self.password}, keepalive=10,
                           client_id="", protocol=4)
            self.last_error = None
            return True
        except Exception as e:  # broker down / auth / DNS: log and carry on
            self.last_error = f"MQTT broker unavailable: {e}"
            logger.warning(self.last_error)
            return False

    def status(self) -> dict:
        return {"enabled": self.enabled, "host": self.host, "port": self.port, "sent": self.sent,
                "last_error": self.last_error, "retry_in_s": round(self.retry_in())}
