"""UI spec v2: the page uses the shared ISTAR kit look, vendored unchanged."""

import hashlib
import socket

from fastapi.testclient import TestClient

from changedetect.config import Settings
from changedetect.web.app import create_app

# sha256 of the ui-kit v2 files; re-copy from the kit and update these together
KIT_VERSION = 2
KIT_SHA256 = {
    "isr-ui.css": "ed601e0494d7b8c93f246205d53a03916b15ceb7eb3a54526a6cd4a447879ec7",
    "isr-ui.js": "191f2b6dce0c52a760a8fdeddf4d5f127399d7d3e1684dcc26a5dbe78874d274",
    "fonts/barlow-condensed-500.woff2": "31ff5e6e96a096fde4d78bc838719e69b093db2d0b062b96e423c065a6d1b2b1",
    "fonts/barlow-condensed-600.woff2": "5bc49d41f32810a114765314392332ce2acae2276635eee265a50319a12c1b49",
    "fonts/ibm-plex-mono-400.woff2": "52de70091213c9eab340e2a7ac59e7b6b2b94e089bf69ef38479b22cfde4274c",
    "fonts/ibm-plex-mono-500.woff2": "f17f51948d10b67d907393ccbac551912f2c2bd2c5af19be3aabe2f456f8fe54",
    "fonts/OFL-BarlowCondensed.txt": "186d750eb496a4c17a76385f82be6aea2ac1cf2de074a811d63786cf374ea73f",
    "fonts/OFL-IBMPlexMono.txt": "7e6b2818edbd8f6a01ae80641cc8f16a51080d08fb4e532be3a0b6f74adb07da",
}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _client(tmp_path):
    st = Settings(data_dir=tmp_path, simulate=True, tak_port=_free_port(), mqtt_port=_free_port(),
                  yolo_model=tmp_path / "models" / "none.onnx", yolo_backend="opencv", vlm_backend="off")
    return TestClient(create_app(st))


def test_page_uses_the_kit(tmp_path):
    html = _client(tmp_path).get("/").text
    assert f'<link rel="stylesheet" href="/static/isr-ui.css?v={KIT_VERSION}">' in html
    assert f'<script src="/static/isr-ui.js?v={KIT_VERSION}"></script>' in html
    assert html.index("isr-ui.css") < html.index("<style>")  # kit before our own CSS
    for needle in ('class="isr-bar"', 'id="isr-status"', 'id="isr-kit-link"', "ISR.setStatus(",
                   "<h1>Change detect</h1>", "Before and after", 'class="isr-btn primary"'):
        assert needle in html
    assert html.count("isr-btn primary") == 1
    assert "Training / demo tool. Verify every change by eye." in html
    assert "fonts.googleapis" not in html and "cdn" not in html.lower()


def test_served_kit_files_are_byte_identical(tmp_path):
    c = _client(tmp_path)
    for name, sha in KIT_SHA256.items():
        r = c.get(f"/static/{name}")
        assert r.status_code == 200
        assert hashlib.sha256(r.content).hexdigest() == sha, f"{name} differs from the kit"
    assert f"v{KIT_VERSION}" in c.get("/static/isr-ui.css").text.splitlines()[0]
