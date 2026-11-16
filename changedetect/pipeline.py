"""End-to-end analysis: inputs -> aligned views -> change blobs -> images, JSON and report."""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from changedetect import align as align_mod
from changedetect import detect as detect_mod
from changedetect import media, render
from changedetect.labeler import Labeler, describe

logger = logging.getLogger(__name__)

VIEWS_360 = [("360 front (0°)", 0.0), ("360 right (90°)", 90.0), ("360 back (180°)", 180.0), ("360 left (270°)", 270.0)]


@dataclass
class Options:
    sensitivity: float = 0.5
    min_area: int = 300
    mode360: str = "auto"  # auto | on | off
    use_yolo: bool = True
    max_dim: int | None = 1600       # flat stills/frames; None = full resolution (laptop)
    max_dim_360: int | None = 4096   # equirectangular panoramas; None = full resolution
    before_time_s: float | None = None
    after_time_s: float | None = None
    pitch_360: float = -15.0
    fov_360: float = 95.0


def new_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def _load_dim(opts: Options) -> int | None:
    if opts.max_dim is None or opts.max_dim_360 is None:
        return None
    return max(opts.max_dim, opts.max_dim_360)


def _load(path: Path, opts: Options, when: float | None):
    kind = media.kind_of(path)
    if kind == "image":
        meta = media.read_exif(path)
        return kind, meta, [(None, media.load_image(path, _load_dim(opts)))]
    meta = media.MediaMeta(name=path.name, kind="video")
    frames = media.sample_video(path, n=5 if when is not None else 12,
                                max_dim=_load_dim(opts), around_s=when)
    meta.height, meta.width = frames[0][1].shape[:2]
    return kind, meta, frames


def _summary(changes: list[dict], views: list[dict]) -> str:
    if not any(v["alignment"]["ok"] for v in views):
        return "ALIGNMENT FAILED - no changes called. Retake from the same position/height/heading."
    if not changes:
        grades = {v["alignment"]["grade"] for v in views if v["alignment"]["ok"]}
        return f"No significant changes found (alignment {'/'.join(sorted(grades))})."
    kinds: dict[str, int] = {}
    for c in changes:
        k = c.get("label_text") or c["kind"]
        kinds[k] = kinds.get(k, 0) + 1
    parts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items(), key=lambda t: -t[1]))
    grades = sorted({v["alignment"]["grade"] for v in views if v["alignment"]["ok"]})
    return f"{len(changes)} change{'s' if len(changes) != 1 else ''}: {parts}. Alignment {'/'.join(grades)}."


def analyze(before_path: Path, after_path: Path, out_root: Path, opts: Options | None = None,
            labeler: Labeler | None = None, result_id: str | None = None) -> dict:
    opts = opts or Options()
    t0 = time.time()
    rid = result_id or new_id()
    out_dir = out_root / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    kb, meta_b, frames_b = _load(Path(before_path), opts, opts.before_time_s)
    ka, meta_a, frames_a = _load(Path(after_path), opts, opts.after_time_s)
    video_info = None
    if len(frames_b) > 1 or len(frames_a) > 1:
        pick = media.best_matching_frames(frames_b, frames_a)
        img_b, img_a = pick["before"], pick["after"]
        if kb == "video":
            meta_b.frame_time_s = pick["before_time_s"]
        if ka == "video":
            meta_a.frame_time_s = pick["after_time_s"]
        video_info = {"before_time_s": pick["before_time_s"], "after_time_s": pick["after_time_s"],
                      "match_inliers": pick["match_inliers"], "frames_compared": [len(frames_b), len(frames_a)]}
    else:
        img_b, img_a = frames_b[0][1], frames_a[0][1]

    is360 = opts.mode360 == "on" or (
        opts.mode360 == "auto" and media.looks_equirect(img_b, meta_b) and media.looks_equirect(img_a, meta_a))
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    yaw_offset = None
    if is360:
        eb = media.resize_max(img_b, opts.max_dim_360)
        ea = media.resize_max(img_a, opts.max_dim_360)
        if ea.shape != eb.shape:
            import cv2
            ea = cv2.resize(ea, (eb.shape[1], eb.shape[0]), interpolation=cv2.INTER_AREA)
        yaw, conf = media.estimate_yaw_offset(eb, ea)
        yaw_offset = {"deg": round(yaw, 1), "confidence": round(conf, 3)}
        out_w = int(np.clip(eb.shape[1] // 4, 960, 2048))
        out_h = out_w * 3 // 4
        for name, base_yaw in VIEWS_360:
            pb = media.equirect_to_perspective(eb, base_yaw, opts.pitch_360, opts.fov_360, out_w, out_h)
            pa = media.equirect_to_perspective(ea, base_yaw + yaw, opts.pitch_360, opts.fov_360, out_w, out_h)
            pairs.append((name, pb, pa))
    else:
        pairs.append(("main", media.resize_max(img_b, opts.max_dim), media.resize_max(img_a, opts.max_dim)))

    use_yolo = bool(opts.use_yolo and labeler is not None and labeler.enabled)
    views, all_changes, view_images = [], [], {}
    counter = 0
    for vi, (name, b, a) in enumerate(pairs):
        prefix = f"v{vi}_"
        al = align_mod.align(b, a)
        view = {"name": name, "alignment": al.to_dict(), "changes": [], "warnings": [], "images": {}}
        imgs: dict[str, np.ndarray] = {}
        render.write_jpeg(out_dir / f"{prefix}before.jpg", b)
        view["images"]["before"] = f"{prefix}before.jpg"
        if not al.ok:
            render.write_jpeg(out_dir / f"{prefix}after.jpg", a)
            view["images"]["after"] = f"{prefix}after.jpg"
            view["images"]["after_aligned"] = None
            imgs["side"] = render.side_by_side(b, a, ("BEFORE", "AFTER (NOT aligned)"))
            render.write_jpeg(out_dir / f"{prefix}side.jpg", imgs["side"])
            view["images"]["side"] = f"{prefix}side.jpg"
            views.append(view)
            view_images[name] = imgs
            continue

        warped, valid = align_mod.warp_after(a, al.homography, b.shape)
        diff = detect_mod.detect_changes(b, warped, valid, opts.sensitivity, opts.min_area)
        view["warnings"] = diff.warnings
        view["threshold"] = round(diff.threshold, 3)
        before_dets = labeler.detect(b) if use_yolo else []
        after_dets = labeler.detect(warped) if use_yolo else []
        for ch in diff.changes:
            counter += 1
            ch.n = counter
            if use_yolo:
                ch.labels, ch.label_text = describe(ch.box, before_dets, after_dets)
            cb, ca = render.crop(b, ch.box), render.crop(warped, ch.box)
            render.write_jpeg(out_dir / f"crop{ch.n}_before.jpg", cb, 92)
            render.write_jpeg(out_dir / f"crop{ch.n}_after.jpg", ca, 92)
            imgs[f"crop{ch.n}_b"], imgs[f"crop{ch.n}_a"] = cb, ca
            d = ch.to_dict()
            d["view"] = name
            d["crops"] = {"before": f"crop{ch.n}_before.jpg", "after": f"crop{ch.n}_after.jpg"}
            view["changes"].append(d)
            all_changes.append(d)

        imgs["heat"] = render.heatmap(b, diff.score_map, diff.valid, diff.threshold)
        bb = render.draw_boxes(b, diff.changes)
        ab = render.draw_boxes(warped, diff.changes)
        imgs["side"] = render.side_by_side(bb, ab)
        for key, img in (("after_aligned", warped), ("heat", imgs["heat"]), ("boxes_before", bb),
                         ("boxes_after", ab), ("side", imgs["side"])):
            render.write_jpeg(out_dir / f"{prefix}{key}.jpg", img)
            view["images"][key] = f"{prefix}{key}.jpg"
        view["image_size"] = [int(b.shape[1]), int(b.shape[0])]
        views.append(view)
        view_images[name] = imgs

    gps = None
    for m, src in ((meta_a, "after image EXIF"), (meta_b, "before image EXIF")):
        if m.has_gps:
            gps = {"lat": m.lat, "lon": m.lon, "alt": m.alt, "source": src}
            break

    result = {
        "id": rid,
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": "360" if is360 else "flat",
        "inputs": {"before": meta_b.to_dict(), "after": meta_a.to_dict()},
        "video": video_info,
        "yaw_offset": yaw_offset,
        "options": {k: v for k, v in opts.__dict__.items()},
        "yolo_used": use_yolo,
        "views": views,
        "changes_total": len(all_changes),
        "alignment_ok": any(v["alignment"]["ok"] for v in views),
        "gps": gps,
        "summary": _summary(all_changes, views),
        "elapsed_s": None,
    }
    result["report"] = render.build_report(result, view_images, out_dir)
    result["elapsed_s"] = round(time.time() - t0, 2)
    save_result(out_dir, result)
    logger.info("Analysis %s done in %.1fs: %s", rid, result["elapsed_s"], result["summary"])
    return result


def save_result(out_dir: Path, result: dict) -> None:
    tmp = out_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out_dir / "result.json")


def rebuild_report(out_dir: Path, result: dict) -> None:
    """Re-render the report after AI suggestions are added (re-reads saved images)."""
    import cv2
    view_images = {}
    for v in result["views"]:
        imgs = {}
        side = v["images"].get("side")
        if side:
            imgs["side"] = cv2.imread(str(out_dir / side))
        if v["images"].get("heat"):
            imgs["heat"] = cv2.imread(str(out_dir / v["images"]["heat"]))
        for ch in v["changes"]:
            imgs[f"crop{ch['n']}_b"] = cv2.imread(str(out_dir / ch["crops"]["before"]))
            imgs[f"crop{ch['n']}_a"] = cv2.imread(str(out_dir / ch["crops"]["after"]))
        view_images[v["name"]] = imgs
    result["report"] = render.build_report(result, view_images, out_dir)
    save_result(out_dir, result)
