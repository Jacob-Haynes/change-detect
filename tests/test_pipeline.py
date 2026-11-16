import json

import cv2
import numpy as np

from changedetect import align, detect, pipeline


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter)


def test_alignment_good_on_synthetic(flat_pair):
    before, after, _ = flat_pair
    al = align.align(before, after)
    assert al.ok and al.grade == "good", al.reasons
    assert al.inliers > 200
    assert al.reproj_error_px < 2.0
    assert al.overlap > 0.8


def test_finds_every_planted_change_without_false_alarms(flat_pair):
    before, after, truth = flat_pair
    al = align.align(before, after)
    warped, valid = align.warp_after(after, al.homography, before.shape)
    res = detect.detect_changes(before, warped, valid)
    found = [c.box for c in res.changes]
    for t in truth:
        assert any(_iou(t["box"], f) > 0.2 for f in found), f"missed {t['what']}"
    false = [f for f in found if not any(_iou(t["box"], f) > 0.05 for t in truth)]
    assert len(false) <= 1, false


def test_identical_images_have_no_changes(flat_pair):
    before, _, _ = flat_pair
    al = align.align(before, before.copy())
    warped, valid = align.warp_after(before, al.homography, before.shape)
    assert detect.detect_changes(before, warped, valid).changes == []


def test_global_lighting_change_is_not_a_change(flat_pair):
    before, _, _ = flat_pair
    darker = np.clip(before.astype(np.float32) * 0.7 + 15, 0, 255).astype(np.uint8)
    al = align.align(before, darker)
    warped, valid = align.warp_after(darker, al.homography, before.shape)
    assert detect.detect_changes(before, warped, valid).changes == []


def test_unrelated_images_refuse_to_call_changes(flat_pair, tmp_path):
    before, _, _ = flat_pair
    rng = np.random.default_rng(3)
    other = cv2.GaussianBlur((rng.random(before.shape) * 255).astype(np.uint8), (0, 0), 4)
    al = align.align(before, other)
    assert not al.ok and al.grade == "failed"
    assert al.reasons

    cv2.imwrite(str(tmp_path / "b.jpg"), before)
    cv2.imwrite(str(tmp_path / "a.png"), other)
    r = pipeline.analyze(tmp_path / "b.jpg", tmp_path / "a.png", tmp_path / "out")
    assert r["alignment_ok"] is False
    assert r["changes_total"] == 0
    assert "FAILED" in r["summary"]


def test_full_pipeline_writes_outputs(samples, tmp_path):
    d, index = samples
    s = index[0]
    r = pipeline.analyze(d / s["before"], d / s["after"], tmp_path)
    out = tmp_path / r["id"]
    assert r["mode"] == "flat" and r["changes_total"] >= 4
    assert r["gps"] and abs(r["gps"]["lat"] - 51.2146) < 0.001
    for name in ("report.png", "report.pdf", "result.json"):
        assert (out / name).stat().st_size > 1000
    v = r["views"][0]
    for key in ("before", "after_aligned", "heat", "boxes_before", "boxes_after", "side"):
        assert (out / v["images"][key]).exists()
    assert json.loads((out / "result.json").read_text())["id"] == r["id"]
    assert (out / "report.pdf").read_bytes()[:4] == b"%PDF"


def test_sensitivity_and_min_area_reduce_changes(flat_pair):
    before, after, _ = flat_pair
    al = align.align(before, after)
    warped, valid = align.warp_after(after, al.homography, before.shape)
    many = detect.detect_changes(before, warped, valid, sensitivity=0.5, min_area=300).changes
    few = detect.detect_changes(before, warped, valid, sensitivity=0.5, min_area=3000).changes
    assert len(few) < len(many)
