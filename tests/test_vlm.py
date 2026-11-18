"""Local VLM call hardening: instruct default, empty content falls back, token cap, no JSON format."""

import http.server
import json
import socket
import threading

import pytest

from changedetect import config, vlm
from changedetect.gemini import GeminiAdvisor


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _ollama(models, reply):
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
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
            self._send({"data": [{"id": m} for m in models]})

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._send(reply)

    port = _free_port()
    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}", seen


class _FakeGemini(GeminiAdvisor):
    def __init__(self):
        super().__init__("fake-key", "gemini-3.8-flash")

    def describe(self, before_jpg, after_jpg, hint=""):
        return "Ground disturbed."


def test_default_model_is_the_instruct_build():
    assert config.DEFAULT_VLM_MODEL == "qwen3-vl:8b-instruct"


@pytest.mark.parametrize("content", ["", None, "   \n ", "<think>the before image shows"])
def test_empty_content_is_a_failure_and_falls_back(content):
    srv, url, seen = _ollama(["qwen3-vl:8b-instruct"],
                             {"choices": [{"message": {"content": content}, "finish_reason": "length"}]})
    try:
        chain = vlm.AdvisorChain("auto", vlm.OpenAICompatAdvisor(url, "qwen3-vl:8b-instruct"), _FakeGemini())
        assert chain.active() == "openai_compat"
        text, source = chain.describe(b"\xff\xd8b", b"\xff\xd8a")
        assert text == "Ground disturbed." and source.startswith("Gemini")
        assert "empty content" in chain.local.last_error and "length" in chain.local.last_error
        assert chain.local.available()  # the endpoint is fine; only that answer was empty
    finally:
        srv.shutdown()


def test_request_caps_tokens_and_asks_for_plain_text():
    srv, url, seen = _ollama(["qwen3-vl:8b-instruct"], {"choices": [{"message": {"content": "Vehicle has left."}}]})
    try:
        adv = vlm.OpenAICompatAdvisor(url, "qwen3-vl:8b-instruct")
        assert adv.describe(b"\xff\xd8b", b"\xff\xd8a") == "Vehicle has left."
        body = seen[-1]
        assert body["max_tokens"] == vlm.MAX_TOKENS <= 128 and body["stream"] is False
        assert "format" not in body and "response_format" not in body  # MLX-runner models: HTTP 501
    finally:
        srv.shutdown()


def test_thinking_build_does_not_stand_in_for_instruct():
    srv, url, _ = _ollama(["qwen3-vl:8b"], {"choices": []})
    try:
        adv = vlm.OpenAICompatAdvisor(url, "qwen3-vl:8b-instruct")
        assert not adv.available() and "ollama pull qwen3-vl:8b-instruct" in adv.last_error
        assert vlm.OpenAICompatAdvisor(url, "qwen3-vl:8b").available()
    finally:
        srv.shutdown()
    srv, url, _ = _ollama(["llava:latest"], {"choices": []})
    try:
        assert vlm.OpenAICompatAdvisor(url, "llava").available()  # untagged name means :latest
    finally:
        srv.shutdown()
