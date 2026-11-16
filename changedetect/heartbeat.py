"""Retained heartbeat on isrkit/heartbeat/change-detect every 10 s, so the Pi launcher can show this
tool wherever it runs (isr-pi or isr-hq). Broker down = log and keep retrying; never fatal.

Payload: {"tool", "host" (short hostname), "url" (http://<host>.local:<port>/), "mode", "status", "ts"}
status is ok | degraded | error while running; "stopped" (retained) after a clean exit; the MQTT
last-will publishes "error" if the process dies without saying goodbye."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

TOOL = "change-detect"
TOPIC = f"isrkit/heartbeat/{TOOL}"
INTERVAL_S = 10


def short_hostname() -> str:
    return socket.gethostname().split(".")[0].lower() or "localhost"


def local_ip(towards: str = "8.8.8.8") -> str:
    """IP of the interface that routes to `towards` (no packets are sent)."""
    try:
        host = towards if towards not in ("127.0.0.1", "localhost") else "8.8.8.8"
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((host, 9))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def build_payload(host: str, url: str, mode: str, status: str, ts: float | None = None) -> dict:
    return {"tool": TOOL, "host": host, "url": url, "mode": mode, "status": status,
            "ts": int(ts if ts is not None else time.time())}


class Heartbeat:
    def __init__(self, mqtt_host: str, mqtt_port: int, user: str, password: str, port: int,
                 mode_fn: Callable[[], str], status_fn: Callable[[], str], public_url: str | None = None,
                 enabled: bool = True):
        self.mqtt_host, self.mqtt_port, self.user, self.password = mqtt_host, mqtt_port, user, password
        self.port, self.mode_fn, self.status_fn, self.public_url = port, mode_fn, status_fn, public_url
        self.enabled = enabled
        self.connected = False
        self.last_error: str | None = None
        self._client = None
        self._stop = threading.Event()

    def url(self) -> str:
        if self.public_url:
            return self.public_url
        return f"http://{short_hostname()}.local:{self.port}/"

    def payload(self, status: str | None = None) -> dict:
        try:
            st = status or self.status_fn()
        except Exception:
            st = "error"
        return build_payload(short_hostname(), self.url(), self.mode_fn(), st)

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Publish a retained "stopped" (if connected) and disconnect. Safe to call twice."""
        self._stop.set()
        t = getattr(self, "_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout)

    def _connect(self):
        import paho.mqtt.client as mqtt
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{TOOL}-{socket.gethostname()}"[:23])
        c.username_pw_set(self.user, self.password)
        c.will_set(TOPIC, json.dumps(self.payload("error")), qos=1, retain=True)

        def on_connect(client, userdata, flags, reason_code, properties=None):
            self.connected = not getattr(reason_code, "is_failure", False)
            if not self.connected:
                self.last_error = f"MQTT refused: {reason_code}"
            elif not self._stop.is_set():
                client.publish(TOPIC, json.dumps(self.payload()), qos=0, retain=True)  # announce at once

        def on_disconnect(client, userdata, flags, reason_code, properties=None):
            self.connected = False

        c.on_connect, c.on_disconnect = on_connect, on_disconnect
        c.reconnect_delay_set(min_delay=2, max_delay=60)
        c.connect_async(self.mqtt_host, self.mqtt_port, keepalive=30)
        c.loop_start()
        return c

    def _run(self) -> None:
        backoff = 2
        while not self._stop.is_set():
            try:
                if self._client is None:
                    self._client = self._connect()
                if self.connected:
                    self._client.publish(TOPIC, json.dumps(self.payload()), qos=0, retain=True)
                    self.last_error = None
                    backoff = 2
            except Exception as e:
                self.last_error = f"heartbeat: {e}"
                logger.debug(self.last_error)
                if self._client is not None:
                    try:
                        self._client.loop_stop()
                    except Exception:
                        pass
                    self._client = None
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60)
                continue
            self._stop.wait(INTERVAL_S)
        if self._client is not None:
            try:
                if self.connected:
                    info = self._client.publish(TOPIC, json.dumps(self.payload("stopped")), qos=1, retain=True)
                    info.wait_for_publish(timeout=2)
                self._client.disconnect()
                self._client.loop_stop()
            except Exception:
                pass

    def status(self) -> dict:
        return {"enabled": self.enabled, "topic": TOPIC, "connected": self.connected, "last_error": self.last_error}
