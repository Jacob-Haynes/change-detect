"""AI suggestion backends: local OpenAI-compatible VLM (Ollama on the HQ Mac) -> Gemini -> off.

VLM_BACKEND=auto (default) tries openai_compat first, then gemini if GEMINI_API_KEY is set.
VLM_BACKEND=openai_compat|gemini|off forces one. The active source is shown in the UI.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request

from changedetect.gemini import PROMPT, GeminiAdvisor

logger = logging.getLogger(__name__)

# One line of <= 20 words needs far fewer; the cap keeps a model that starts reasoning from running on.
MAX_TOKENS = 96
DEFAULT_TIMEOUT_S = 30.0  # qwen3-vl:8b-instruct answers in 1-5 s; fail over to the next backend quickly


def _with_tag(model: str) -> str:
    return model if ":" in model else model + ":latest"


def base_url(url: str) -> str:
    """Normalise to the bare Ollama base URL (tolerates a trailing / or /v1)."""
    u = (url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3]
    return u


class OpenAICompatAdvisor:
    name = "openai_compat"

    def __init__(self, url: str, model: str, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT_S):
        self.url = base_url(url)
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.last_error: str | None = None
        self._reach: tuple[float, bool] = (0.0, False)

    def _req(self, path: str, body: dict | None = None, timeout: float = 3.0):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method="POST" if data else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def available(self) -> bool:
        """Endpoint reachable and the model is listed (cached 30 s)."""
        t, ok = self._reach
        if time.time() - t < 30:
            return ok
        try:
            models = [m.get("id") for m in self._req("/v1/models").get("data", [])]
            # exact tag match: qwen3-vl:8b (a thinking build) must not stand in for qwen3-vl:8b-instruct
            ok = _with_tag(self.model) in {_with_tag(m) for m in models if m}
            self.last_error = None if ok else f"model {self.model} not pulled (ollama pull {self.model})"
        except Exception as e:
            ok = False
            self.last_error = f"VLM endpoint {self.url} unreachable: {e}"
        self._reach = (time.time(), ok)
        return ok

    def describe(self, before_jpg: bytes, after_jpg: bytes, hint: str = "") -> str | None:
        def img(b):
            return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}
        # Plain text out: no Ollama "format":"json" / response_format (MLX-runner models answer HTTP 501).
        body = {"model": self.model, "temperature": 0.2, "max_tokens": MAX_TOKENS, "stream": False,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT + (f" Detector's own guess: {hint}." if hint else "")},
                    img(before_jpg), img(after_jpg)]}]}
        try:
            out = self._req("/v1/chat/completions", body, timeout=self.timeout)
        except Exception as e:
            self.last_error = f"VLM request failed: {e}"
            logger.warning(self.last_error)
            self._reach = (time.time(), False)
            return None
        try:
            choice = out["choices"][0]
            text = (choice["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            choice, text = {}, ""
        # thinking builds may wrap their reasoning; keep what follows it
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        elif text.startswith("<think>"):
            text = ""  # cut off mid-thought by the token cap
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:  # empty content is a failure, so the chain falls back to the next backend
            why = choice.get("finish_reason") if isinstance(choice, dict) else None
            self.last_error = (f"VLM {self.model} returned empty content" + (f" (finish_reason {why})" if why else "")
                               + "; use an instruct build such as qwen3-vl:8b-instruct")
            logger.warning(self.last_error)
            return None
        self.last_error = None
        return lines[-1][:200]


class AdvisorChain:
    def __init__(self, backend: str, local: OpenAICompatAdvisor, gemini: GeminiAdvisor):
        self.backend = (backend or "auto").lower()
        self.local, self.gemini = local, gemini
        self.last_source: str | None = None

    def _order(self):
        if self.backend == "off":
            return []
        if self.backend == "gemini":
            return [("gemini", self.gemini)] if self.gemini.enabled else []
        order = [("openai_compat", self.local)]
        if self.backend == "auto" and self.gemini.enabled:
            order.append(("gemini", self.gemini))
        return order

    def active(self) -> str:
        for name, adv in self._order():
            if name == "gemini" or adv.available():
                return name
        return "off"

    @property
    def enabled(self) -> bool:
        return self.active() != "off"

    def source_label(self, name: str | None = None) -> str:
        name = name or self.active()
        if name == "openai_compat":
            return f"local VLM ({self.local.model})"
        if name == "gemini":
            return f"Gemini cloud ({self.gemini.model})"
        return "off"

    def describe(self, before_jpg: bytes, after_jpg: bytes, hint: str = "") -> tuple[str | None, str | None]:
        for name, adv in self._order():
            if name != "gemini" and not adv.available():
                continue
            text = adv.describe(before_jpg, after_jpg, hint)
            if text:
                self.last_source = name
                return text, self.source_label(name)
        return None, None

    @property
    def last_error(self) -> str | None:
        return self.local.last_error or self.gemini.last_error

    def status(self) -> dict:
        act = self.active()
        return {"backend": self.backend, "active": act, "source": self.source_label(act),
                "enabled": act != "off", "cloud": act == "gemini",
                "local": {"url": self.local.url, "model": self.local.model, "error": self.local.last_error},
                "gemini": self.gemini.status(),
                # kept for older UI code
                "model": self.gemini.model if act == "gemini" else (self.local.model if act != "off" else None)}
