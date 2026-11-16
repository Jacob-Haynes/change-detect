"""Folder / SD-card batches: split a card dump into two passes, pair matching shots, analyse each pair.

Typical use: drag the DCIM folder (or two folders, one per pass) into the UI. Without explicit
before/after folders the images are sorted by capture time (EXIF, else file modified time sent
by the browser) and split at the largest time gap: the first pass is BEFORE, the second AFTER.
"""

from __future__ import annotations

import json
import logging
import math
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from changedetect import media, pipeline

logger = logging.getLogger(__name__)

MIN_PAIR_INLIERS = 20
GPS_RADIUS_M = 40.0


@dataclass
class Item:
    path: Path
    name: str
    ts: float | None = None
    lat: float | None = None
    lon: float | None = None
    feats: object = None
    meta: dict = field(default_factory=dict)


def _exif_ts(t: str | None) -> float | None:
    if not t:
        return None
    try:
        return datetime.fromisoformat(t[:19]).timestamp()
    except ValueError:
        return None


def make_items(paths: list[Path], mtimes: dict[str, float] | None = None) -> list[Item]:
    items = []
    for p in paths:
        if media.kind_of(p) != "image":
            continue
        m = media.read_exif(p)
        orig = p.name.split("_", 1)[1] if "_" in p.name else p.name
        ts = _exif_ts(m.time)
        if ts is None and mtimes:
            ms = mtimes.get(orig)
            ts = ms / 1000.0 if ms else None
        items.append(Item(path=p, name=orig, ts=ts, lat=m.lat, lon=m.lon, meta=m.to_dict()))
    return items


def split_by_time(items: list[Item]) -> tuple[list[Item], list[Item], dict]:
    """Split at the largest gap in capture time. Raises ValueError if that is not possible."""
    timed = [i for i in items if i.ts is not None]
    if len(timed) < 2 or len(timed) < 0.8 * len(items):
        raise ValueError("cannot auto-split: capture times missing. Drop the BEFORE and AFTER passes "
                         "into their own boxes instead.")
    timed.sort(key=lambda i: (i.ts, i.name))
    gaps = [(timed[k + 1].ts - timed[k].ts, k) for k in range(len(timed) - 1)]
    gap, k = max(gaps)
    others = sorted(g for g, _ in gaps if _ != k)
    median_gap = others[len(others) // 2] if others else 0.0
    if gap < 120 or (median_gap and gap < 5 * median_gap):
        raise ValueError(f"cannot auto-split: no clear break between two passes (largest gap {gap:.0f}s). "
                         "Drop each pass into its own box.")
    info = {"split_gap_s": round(gap, 1), "split_after": timed[k].name, "split_before": timed[k + 1].name}
    return timed[: k + 1], timed[k + 1:], info


def _dist_m(a: Item, b: Item) -> float:
    lat = math.radians((a.lat + b.lat) / 2)
    dx = (b.lon - a.lon) * 111_320 * math.cos(lat)
    dy = (b.lat - a.lat) * 110_540
    return math.hypot(dx, dy)


def _feats(item: Item):
    if item.feats is None:
        item.feats = media._quick_features(media.load_image(item.path, 640))
    return item.feats


def pair_items(before: list[Item], after: list[Item]) -> tuple[list[dict], list[str], list[str]]:
    """Greedy one-to-one pairing by RANSAC inliers on thumbnails, pruned by GPS (if both have it)
    or by position in the sequence (both passes follow the same route)."""
    before = sorted(before, key=lambda i: (i.ts or 0, i.name))
    after = sorted(after, key=lambda i: (i.ts or 0, i.name))
    nb, na = len(before), len(after)
    window = max(6, int(0.15 * max(nb, na)))
    scores = []
    for bi, b in enumerate(before):
        if b.lat is not None and any(a.lat is not None for a in after):
            cands = [(ai, a) for ai, a in enumerate(after) if a.lat is not None and _dist_m(b, a) <= GPS_RADIUS_M]
        else:
            centre = bi * (na / max(nb, 1))
            cands = [(ai, a) for ai, a in enumerate(after) if abs(ai - centre) <= window]
        for ai, a in cands:
            s = media._quick_inliers(_feats(b), _feats(a))
            if s >= MIN_PAIR_INLIERS:
                scores.append((s, bi, ai))
    scores.sort(reverse=True)
    used_b, used_a, pairs = set(), set(), []
    for s, bi, ai in scores:
        if bi in used_b or ai in used_a:
            continue
        used_b.add(bi)
        used_a.add(ai)
        pairs.append({"before": before[bi].name, "after": after[ai].name, "match_inliers": s,
                      "_b": before[bi].path, "_a": after[ai].path, "_order": bi})
    pairs.sort(key=lambda p: p["_order"])
    unpaired_b = [before[i].name for i in range(nb) if i not in used_b]
    unpaired_a = [after[i].name for i in range(na) if i not in used_a]
    return pairs, unpaired_b, unpaired_a


class BatchRunner:
    def __init__(self, batches_dir: Path, reports_dir: Path, lock: threading.Lock, labeler=None):
        self.batches_dir, self.reports_dir, self.lock, self.labeler = batches_dir, reports_dir, lock, labeler
        self.jobs: dict[str, dict] = {}

    @staticmethod
    def new_id() -> str:
        return "B" + datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)

    def _save(self, job: dict) -> None:
        public = {k: v for k, v in job.items() if not k.startswith("_")}
        public["pairs"] = [{k: v for k, v in p.items() if not k.startswith("_")} for p in job.get("pairs", [])]
        (self.batches_dir / f"{job['id']}.json").write_text(json.dumps(public, indent=2))

    def get(self, bid: str) -> dict | None:
        if bid in self.jobs:
            job = self.jobs[bid]
            out = {k: v for k, v in job.items() if not k.startswith("_")}
            out["pairs"] = [{k: v for k, v in p.items() if not k.startswith("_")} for p in job.get("pairs", [])]
            return out
        f = self.batches_dir / f"{bid}.json"
        return json.loads(f.read_text()) if f.exists() else None

    def list(self, limit: int = 20) -> list[dict]:
        files = sorted(self.batches_dir.glob("B*.json"), reverse=True)[:limit]
        out = []
        for f in files:
            try:
                j = json.loads(f.read_text())
            except ValueError:
                continue
            out.append({k: j.get(k) for k in ("id", "created", "state", "summary")})
        return out

    def start(self, upload_dir: Path, before: list[Path] | None, after: list[Path] | None,
              all_files: list[Path] | None, mtimes: dict, opts: pipeline.Options, bid: str) -> dict:
        job = {"id": bid, "created": datetime.now().isoformat(timespec="seconds"), "state": "pairing",
               "progress": 0, "total": 0, "pairs": [], "unpaired_before": [], "unpaired_after": [],
               "split": None, "error": None, "summary": "pairing images…", "_dir": upload_dir}
        self.jobs[bid] = job
        self._save(job)
        threading.Thread(target=self._run, args=(job, before, after, all_files, mtimes, opts),
                         name=f"batch-{bid}", daemon=True).start()
        return self.get(bid)

    def _run(self, job, before, after, all_files, mtimes, opts):
        t0 = time.time()
        try:
            if all_files:
                items = make_items(all_files, mtimes)
                b_items, a_items, job["split"] = split_by_time(items)
            else:
                b_items, a_items = make_items(before, mtimes), make_items(after, mtimes)
            if not b_items or not a_items:
                raise ValueError("need at least one still image in each pass (JPEG/HEIC/PNG)")
            pairs, ub, ua = pair_items(b_items, a_items)
            job.update(pairs=pairs, unpaired_before=ub, unpaired_after=ua, total=len(pairs), state="analysing")
            if not pairs:
                raise ValueError("no matching shots found between the two passes; retake from the same points")
            self._save(job)
            for i, p in enumerate(pairs):
                job["summary"] = f"analysing pair {i + 1}/{len(pairs)}"
                with self.lock:
                    try:
                        r = pipeline.analyze(p["_b"], p["_a"], self.reports_dir, opts, self.labeler)
                        p.update(result_id=r["id"], summary=r["summary"], changes=r["changes_total"],
                                 alignment_ok=r["alignment_ok"],
                                 grade="/".join(sorted({v["alignment"]["grade"] for v in r["views"]})),
                                 thumb=r["views"][0]["images"].get("boxes_before") or r["views"][0]["images"]["before"])
                        out = self.reports_dir / r["id"]
                        for src, tag in ((p["_b"], "before"), (p["_a"], "after")):
                            try:
                                (out / f"original_{tag}{src.suffix.lower()}").write_bytes(src.read_bytes())
                            except OSError:
                                pass
                    except Exception as e:  # one bad pair must not stop the batch
                        logger.exception("batch pair failed")
                        p.update(error=str(e), alignment_ok=False, changes=0)
                job["progress"] = i + 1
                self._save(job)
            total = sum(p.get("changes", 0) for p in pairs)
            failed = sum(1 for p in pairs if not p.get("alignment_ok"))
            job["state"] = "done"
            job["summary"] = (f"{len(pairs)} pairs, {total} changes"
                              + (f", {failed} pairs not aligned" if failed else "")
                              + (f", {len(ub) + len(ua)} unpaired" if ub or ua else ""))
        except Exception as e:
            job["state"], job["error"], job["summary"] = "error", str(e), f"error: {e}"
        finally:
            job["elapsed_s"] = round(time.time() - t0, 1)
            self._save(job)
            d = job.get("_dir")
            if d:
                shutil.rmtree(d, ignore_errors=True)
