"""CoT change-marker XML. Delivery to the TAK server is changedetect.tak_client.TakClient."""

from __future__ import annotations

import time
from xml.sax.saxutils import escape, quoteattr

from changedetect.tak_client import iso as _iso

COT_TYPE = "a-u-G"  # unknown ground point: a change, not an identification
STALE_SECONDS = 6 * 3600


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
