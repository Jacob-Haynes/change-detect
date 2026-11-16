"""Publish short mesh texts via the meshbus MQTT topic (the blufor-tracker bridge owns the radio)."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

TX_TOPIC = "meshbus/tx"
MAX_LEN = 200


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

    def send(self, text: str, to: str | None = None, channel: int = 0) -> bool:
        if not self.enabled:
            return False
        payload = build_payload(text, to, channel)
        try:
            from paho.mqtt import publish
            publish.single(TX_TOPIC, json.dumps(payload), qos=1, hostname=self.host, port=self.port,
                           auth={"username": self.user, "password": self.password}, keepalive=10,
                           client_id="", protocol=4)
            self.sent += 1
            self.last_error = None
            return True
        except Exception as e:  # broker down / auth / DNS: log and carry on
            self.last_error = f"MQTT broker unavailable: {e}"
            logger.warning(self.last_error)
            return False

    def status(self) -> dict:
        return {"enabled": self.enabled, "host": self.host, "port": self.port, "sent": self.sent,
                "last_error": self.last_error}
