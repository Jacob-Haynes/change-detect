"""CoT change-marker XML. Delivery to the TAK server is changedetect.tak_client.TakClient.

Markers are built so OpenTAKServer 1.8.3 both relays and STORES them (contract C5.5):
type a-u-G matches ^a-[fhupansjk]-[ZPAGSUF]; no <takv> and no <contact endpoint>; a real
position (never 0,0); and the only <link uid> is our own registration (EUD) uid.
"""

from __future__ import annotations

import time
from xml.sax.saxutils import escape, quoteattr

from changedetect.tak_client import iso as _iso

COT_TYPE = "a-u-G"  # unknown ground point: a change, not an identification
STALE_SECONDS = 6 * 3600


def build_change_cot(uid: str, lat: float, lon: float, callsign: str, remarks: str,
                     manual: bool = False, ce: float = 25.0, now: float | None = None,
                     parent_uid: str | None = None, parent_callsign: str | None = None) -> str:
    if lat is None or lon is None or (lat == 0 and lon == 0) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("a change marker needs a real position (never 0,0)")
    now = time.time() if now is None else now
    how = "h-e" if manual else "m-g"
    link = ""
    if parent_uid:  # must be an EUD uid (OTS markers.parent_uid -> euds.uid), i.e. our registration
        link = (f'<link uid={quoteattr(parent_uid)} relation="p-p" type="a-f-G-E-S" production_time="{_iso(now)}"'
                + (f" parent_callsign={quoteattr(parent_callsign)}" if parent_callsign else "") + "/>")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(uid)} type="{COT_TYPE}" how="{how}" '
        f'time="{_iso(now)}" start="{_iso(now)}" stale="{_iso(now + STALE_SECONDS)}">'
        f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="9999999.0" ce="{ce:.1f}" le="9999999.0"/>'
        f'<detail><contact callsign={quoteattr(callsign)}/>'
        f'<remarks>{escape(remarks)}</remarks>{link}'
        f'<__group name="Yellow" role="Team Member"/></detail></event>'
    )
