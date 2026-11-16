"""Synthetic before/after pairs for --simulate mode and tests.

Draws a textured overhead scene (fields, track, buildings, trees, a car), then makes an
AFTER version where the car has moved, a camouflaged hide has appeared and a patch of
ground has been disturbed. Both are rendered from slightly different viewpoints and the
AFTER image gets different lighting and noise, like two real passes over the same area.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCENE_W, SCENE_H = 1800, 1350
OUT_W, OUT_H = 1440, 1080
# Approximate position for the demo marker (Salisbury Plain training area)
DEMO_LAT, DEMO_LON = 51.2146, -1.8079


def _noise(h, w, rng, octaves=(4, 8, 16, 48, 128, 384)):
    acc = np.zeros((h, w), np.float32)
    for i, o in enumerate(octaves):
        small = rng.random((max(2, h * o // 1000), max(2, w * o // 1000))).astype(np.float32)
        acc += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * (0.6 ** i)
    acc -= acc.min()
    return acc / (acc.max() + 1e-6)


def _scene(rng: np.random.Generator, w=SCENE_W, h=SCENE_H) -> np.ndarray:
    n = _noise(h, w, rng)
    grass = np.array([60, 120, 80], np.float32)   # BGR
    dry = np.array([90, 150, 150], np.float32)
    img = grass * (1 - n[..., None]) + dry * n[..., None]
    img += (rng.random((h, w, 1)).astype(np.float32) - 0.5) * 30
    # field boundaries / hedgerows
    for _ in range(6):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, (35, 75, 40), int(rng.integers(6, 14)))
    # dirt track
    pts = np.array([[0, 300], [400, 420], [800, 380], [1200, 620], [1800, 700]], np.int32)
    cv2.polylines(img, [pts], False, (110, 150, 170), 34)
    cv2.polylines(img, [pts], False, (95, 135, 160), 4)
    # buildings with roof detail
    for (x, y, bw, bh) in [(1350, 150, 220, 140), (1100, 900, 160, 200), (250, 800, 260, 150)]:
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (120, 120, 125), -1)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (60, 60, 70), 4)
        for k in range(1, 5):
            cv2.line(img, (x + k * bw // 5, y), (x + k * bw // 5, y + bh), (95, 95, 105), 2)
    # trees
    for _ in range(90):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        r = int(rng.integers(8, 26))
        cv2.circle(img, c, r, (25, 65, 30), -1)
        cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), max(2, r // 3), (50, 105, 55), -1)
    # scattered rocks / clutter for features
    for _ in range(400):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        v = float(rng.integers(140, 220))
        cv2.circle(img, c, int(rng.integers(1, 4)), (v, v, v), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _car(img, cx, cy, angle, colour=(140, 50, 30)):
    rect = ((cx, cy), (64, 30), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(img, [box], colour)
    cv2.polylines(img, [box], True, (20, 20, 20), 2)
    inner = cv2.boxPoints(((cx, cy), (30, 22), angle)).astype(np.int32)
    cv2.fillPoly(img, [inner], (200, 190, 170))


def _hide(img, cx, cy, rng):
    poly = np.array([[cx + 50 * np.cos(t) * (0.8 + 0.4 * rng.random()),
                      cy + 35 * np.sin(t) * (0.8 + 0.4 * rng.random())]
                     for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)], np.int32)
    cv2.fillPoly(img, [poly], (30, 55, 40))
    x0, y0 = cx - 55, cy - 40
    for k in range(0, 110, 8):  # net pattern
        cv2.line(img, (x0 + k, y0), (x0 + k - 40, y0 + 80), (55, 85, 60), 1)
    cv2.polylines(img, [poly], True, (20, 40, 25), 2)


def _disturbed(img, cx, cy, rng):
    h, w = 60, 90
    patch = (rng.random((h, w)) * 50 + 70).astype(np.float32)
    patch = cv2.GaussianBlur(patch, (5, 5), 0)
    colour = np.stack([patch * 0.55, patch * 0.8, patch * 1.05], -1)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 15, 0, 360, 255, -1)
    roi = img[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2]
    roi[mask > 0] = np.clip(colour[mask > 0], 0, 255).astype(np.uint8)


def _view(scene, H, size=(OUT_W, OUT_H)):
    return cv2.warpPerspective(scene, H, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _homography(tx, ty, rot_deg, scale, persp=(0.0, 0.0)):
    cx, cy = SCENE_W / 2, SCENE_H / 2
    a = np.radians(rot_deg)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    R = np.array([[scale * np.cos(a), -scale * np.sin(a), 0], [scale * np.sin(a), scale * np.cos(a), 0], [0, 0, 1]])
    P = np.array([[1, 0, 0], [0, 1, 0], [persp[0], persp[1], 1]], float)
    T2 = np.array([[1, 0, OUT_W / 2 + tx], [0, 1, OUT_H / 2 + ty], [0, 0, 1]], float)
    return T2 @ P @ R @ T1


def make_pair(seed: int = 7):
    """Return (before_bgr, after_bgr, ground_truth) with boxes in BEFORE pixel coords."""
    rng = np.random.default_rng(seed)
    scene = _scene(rng)
    car_a, car_b = (700, 560), (1020, 470)
    hide_c, dig_c = (560, 1010), (1300, 560)
    before_scene = scene.copy()
    _car(before_scene, *car_a, angle=20)
    after_scene = scene.copy()
    _car(after_scene, *car_b, angle=-15)
    _hide(after_scene, *hide_c, rng)
    _disturbed(after_scene, *dig_c, rng)

    Hb = _homography(0, 0, 0.0, 0.9)
    Ha = _homography(24, -18, 3.5, 0.93, persp=(1.5e-5, -1e-5))
    before = _view(before_scene, Hb)
    after = _view(after_scene, Ha).astype(np.float32)
    # different light: darker, warmer, slight gradient, sensor noise
    gx = np.linspace(0.92, 1.05, OUT_W, dtype=np.float32)[None, :, None]
    after = after * 0.82 * gx + np.array([4, 10, 18], np.float32)
    after += rng.normal(0, 3.0, after.shape).astype(np.float32)
    after = np.clip(after, 0, 255).astype(np.uint8)
    before = np.clip(before.astype(np.float32) + rng.normal(0, 2.0, before.shape), 0, 255).astype(np.uint8)

    def box_in_before(c, half):
        pts = np.float32([[c[0] - half[0], c[1] - half[1]], [c[0] + half[0], c[1] + half[1]],
                          [c[0] - half[0], c[1] + half[1]], [c[0] + half[0], c[1] - half[1]]]).reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(pts, Hb).reshape(-1, 2)
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

    truth = [
        {"what": "vehicle gone", "box": box_in_before(car_a, (40, 32))},
        {"what": "vehicle appeared", "box": box_in_before(car_b, (40, 32))},
        {"what": "new hide", "box": box_in_before(hide_c, (58, 42))},
        {"what": "disturbed ground", "box": box_in_before(dig_c, (46, 31))},
    ]
    return before, after, truth


def make_equirect_pair(seed: int = 11, yaw_offset_deg: float = 37.0, w: int = 3072):
    """Synthetic 360 pair: AFTER is rotated in heading and has two changes."""
    rng = np.random.default_rng(seed)
    h = w // 2
    base = _scene(rng, w, h)
    # sky band so it looks like a panorama
    sky = np.linspace(230, 170, h // 4, dtype=np.float32)[:, None, None] * np.array([1.0, 0.85, 0.7])
    base[: h // 4] = np.clip(sky + rng.normal(0, 2, (h // 4, w, 3)), 0, 255).astype(np.uint8)
    before = base.copy()
    _car(before, int(w * 0.25), int(h * 0.6), 0)
    after = base.copy()
    _hide(after, int(w * 0.75), int(h * 0.62), rng)
    shift = int(round(yaw_offset_deg / 360.0 * w))
    after = np.roll(after, shift, axis=1)
    after = np.clip(after.astype(np.float32) * 0.88 + 8 + rng.normal(0, 2.5, after.shape), 0, 255).astype(np.uint8)
    truth = {"yaw_offset_deg": yaw_offset_deg,
             "changes": [{"what": "vehicle gone", "yaw_deg": -90.0}, {"what": "new hide", "yaw_deg": 90.0}]}
    return before, after, truth


def _save_jpeg_with_gps(path: Path, bgr: np.ndarray, lat: float, lon: float, when: str, equirect=False):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    exif[0x010F] = "SYNTHETIC"
    exif[0x0110] = "change-detect demo"
    exif[0x0132] = when

    def dms(v):
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        s = round(((v - d) * 60 - m) * 60, 4)
        return (float(d), float(m), float(s))

    gps = exif.get_ifd(0x8825)
    gps[1] = "N" if lat >= 0 else "S"
    gps[2] = dms(lat)
    gps[3] = "E" if lon >= 0 else "W"
    gps[4] = dms(lon)
    kwargs = {"exif": exif, "quality": 92}
    if equirect:
        kwargs["xmp"] = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                         b'<rdf:Description xmlns:GPano="http://ns.google.com/photos/1.0/panorama/" '
                         b'GPano:ProjectionType="equirectangular"/></rdf:RDF></x:xmpmeta>')
    im.save(path, "JPEG", **kwargs)


def write_samples(samples_dir: Path, force: bool = False) -> list[dict]:
    """Write bundled synthetic samples (idempotent). Returns the sample index."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    index_path = samples_dir / "index.json"
    if index_path.exists() and not force:
        try:
            return json.loads(index_path.read_text())
        except ValueError:
            pass
    before, after, truth = make_pair()
    _save_jpeg_with_gps(samples_dir / "synthetic_before.jpg", before, DEMO_LAT, DEMO_LON, "2026:11:14 09:12:30")
    _save_jpeg_with_gps(samples_dir / "synthetic_after.jpg", after, DEMO_LAT + 0.00002, DEMO_LON - 0.00003,
                        "2026:11:14 15:47:05")
    eb, ea, etruth = make_equirect_pair()
    _save_jpeg_with_gps(samples_dir / "synthetic360_before.jpg", eb, DEMO_LAT + 0.001, DEMO_LON + 0.002,
                        "2026:11:14 09:30:00", equirect=True)
    _save_jpeg_with_gps(samples_dir / "synthetic360_after.jpg", ea, DEMO_LAT + 0.001, DEMO_LON + 0.002,
                        "2026:11:14 16:02:00", equirect=True)
    index = [
        {"id": "synthetic", "title": "Synthetic overhead pass (car moved, new hide, disturbed ground)",
         "before": "synthetic_before.jpg", "after": "synthetic_after.jpg", "truth": truth},
        {"id": "synthetic360", "title": "Synthetic 360 still (heading changed 37°, car gone, new hide)",
         "before": "synthetic360_before.jpg", "after": "synthetic360_after.jpg", "truth": etruth},
    ]
    index_path.write_text(json.dumps(index, indent=2))
    return index
