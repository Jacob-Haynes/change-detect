"""Loading stills and videos, EXIF GPS/time, and 360 equirectangular helpers."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

try:  # iPhone HEIC/HEIF stills
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_OK = True
except Exception:  # pragma: no cover - optional dependency
    HEIF_OK = False

# .insp = Insta360 ONE still (a JPEG of two side-by-side fisheye circles); .insv = its MP4 video
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif", ".insp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".insv"}
DUAL_FISHEYE_EXT = {".insp", ".insv"}
# Insta360 ONE lenses are ~210 deg; 200 is a safe usable field per lens. Override with FISHEYE_FOV_DEG.
DEFAULT_FISHEYE_FOV = 200.0

# EXIF tag ids
_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_MAKE, _MODEL, _DATETIME = 0x010F, 0x0110, 0x0132
_DT_ORIGINAL, _OFFSET_ORIGINAL = 0x9003, 0x9011


class MediaError(Exception):
    pass


@dataclass
class MediaMeta:
    name: str = ""
    kind: str = "image"  # image | video
    width: int = 0
    height: int = 0
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    time: str | None = None
    make: str | None = None
    model: str | None = None
    equirect_tag: bool = False
    frame_time_s: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None


def kind_of(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".heic", ".heif") and not HEIF_OK:
        raise MediaError("HEIC support needs the pillow-heif package (pip install pillow-heif).")
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    raise MediaError(f"Unsupported file type '{ext}'. Use JPEG/HEIC/PNG stills or MP4/MOV video.")


def _rational(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        num, den = v
        return float(num) / float(den) if den else 0.0


def _dms_to_deg(dms, ref) -> float | None:
    if not dms:
        return None
    try:
        parts = [_rational(x) for x in dms]
    except Exception:
        return None
    while len(parts) < 3:
        parts.append(0.0)
    deg = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors="ignore")
    if ref and str(ref).strip().upper() in ("S", "W"):
        deg = -deg
    return deg


def _clean(s) -> str | None:
    if s is None:
        return None
    if isinstance(s, bytes):
        s = s.decode(errors="ignore")
    s = str(s).strip("\x00 ").strip()
    return s or None


def read_exif(path: str | Path) -> MediaMeta:
    """Pull GPS, capture time and camera make/model from a still. Missing fields stay None."""
    path = Path(path)
    meta = MediaMeta(name=path.name, kind="image")
    try:
        with Image.open(path) as im:
            meta.width, meta.height = im.size
            exif = im.getexif()
            meta.make = _clean(exif.get(_MAKE))
            meta.model = _clean(exif.get(_MODEL))
            sub = exif.get_ifd(_EXIF_IFD)
            dt = _clean(sub.get(_DT_ORIGINAL) or exif.get(_DATETIME))
            if dt and len(dt) >= 19:
                meta.time = dt[:10].replace(":", "-") + "T" + dt[11:19]
                off = _clean(sub.get(_OFFSET_ORIGINAL))
                if off:
                    meta.time += off
            gps = exif.get_ifd(_GPS_IFD)
            if gps:
                lat = _dms_to_deg(gps.get(2), gps.get(1))
                lon = _dms_to_deg(gps.get(4), gps.get(3))
                if lat is not None and lon is not None and not (lat == 0 and lon == 0):
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        meta.lat, meta.lon = round(lat, 7), round(lon, 7)
                if gps.get(6) is not None:
                    alt = _rational(gps.get(6))
                    ref = gps.get(5)
                    if ref in (1, b"\x01"):
                        alt = -alt
                    meta.alt = round(alt, 1)
    except Exception as e:  # corrupt EXIF must never stop an analysis
        logger.info("EXIF read failed for %s: %s", path.name, e)
    # 360 cameras tag equirectangular stills in XMP (GPano:ProjectionType)
    try:
        with open(path, "rb") as f:
            head = f.read(512 * 1024)
        meta.equirect_tag = b"equirectangular" in head.lower()
    except OSError:
        pass
    return meta


def load_image(path: str | Path, max_dim: int | None = None) -> np.ndarray:
    """Load a still as BGR uint8, honouring EXIF orientation, optionally downscaled."""
    try:
        with Image.open(path) as im:
            if max_dim and im.format == "JPEG":
                im.draft("RGB", (max_dim, max_dim))  # fast DCT downscale for big GoPro/drone stills
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"))
    except Exception as e:
        raise MediaError(f"Could not read image {Path(path).name}: {e}") from e
    return resize_max(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), max_dim)


def resize_max(img: np.ndarray, max_dim: int | None) -> np.ndarray:
    if not max_dim:
        return img
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def looks_equirect(img: np.ndarray, meta: MediaMeta | None = None) -> bool:
    if meta is not None and meta.equirect_tag:
        return True
    h, w = img.shape[:2]
    return w >= 1500 and abs(w / h - 2.0) < 0.02 and not looks_dual_fisheye(img)


# ---------------------------------------------------------------- dual fisheye (Insta360 ONE)


def looks_dual_fisheye(img: np.ndarray) -> bool:
    """Two fisheye circles side by side: ~2:1 frame, each half has dark corners and a bright middle."""
    h, w = img.shape[:2]
    if h < 200 or abs(w / h - 2.0) > 0.05:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    k = max(4, h // 20)
    for x0 in (0, w // 2):
        half = gray[:, x0:x0 + w // 2]
        hh, hw = half.shape
        corners = np.concatenate([half[:k, :k].ravel(), half[:k, -k:].ravel(),
                                  half[-k:, :k].ravel(), half[-k:, -k:].ravel()])
        centre = half[hh // 2 - 2 * k:hh // 2 + 2 * k, hw // 2 - 2 * k:hw // 2 + 2 * k]
        if np.median(corners) > 25 or centre.mean() < np.median(corners) + 20:
            return False
    return True


def _circle(half_gray: np.ndarray) -> tuple[float, float, float]:
    """Centre and radius of the lit fisheye circle in one half (falls back to the inscribed circle)."""
    h, w = half_gray.shape
    mask = half_gray > 18
    ys, xs = np.nonzero(mask)
    if len(xs) < 0.3 * h * w * np.pi / 4:
        return w / 2, h / 2, min(w, h) / 2
    x0, x1 = np.percentile(xs, [0.2, 99.8])
    y0, y1 = np.percentile(ys, [0.2, 99.8])
    r = max(x1 - x0, y1 - y0) / 2
    return (x0 + x1) / 2, (y0 + y1) / 2, min(r, min(w, h) / 2 * 1.02)


def dual_fisheye_to_equirect(img: np.ndarray, fov_deg: float = DEFAULT_FISHEYE_FOV,
                             out_w: int | None = None) -> np.ndarray:
    """Unwarp a side-by-side dual-fisheye frame to an equirectangular panorama.

    Equidistant lens model, left circle = front (+z), right circle = back (-z), no seam blending.
    Not a calibrated stitch: before and after go through the same mapping, so the comparison
    stays consistent even if the result is slightly off true geometry.
    """
    h, w = img.shape[:2]
    half = w // 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    circles = [_circle(gray[:, :half]), _circle(gray[:, half:2 * half])]
    out_w = out_w or min(2 * half, 6144)
    out_h = out_w // 2
    lon = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(out_h, dtype=np.float32) + 0.5) / out_h * np.pi
    lon, lat = np.meshgrid(lon, lat)
    x = np.cos(lat) * np.sin(lon)
    y = -np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    half_fov = np.radians(fov_deg) / 2
    front = z >= 0
    map_x = np.empty_like(x)
    map_y = np.empty_like(x)
    for is_front, (cx, cy, r), off in ((True, circles[0], 0), (False, circles[1], half)):
        sel = front if is_front else ~front
        xx = x if is_front else -x
        zz = z if is_front else -z
        theta = np.arccos(np.clip(zz, -1, 1))
        phi = np.arctan2(y, xx)
        rr = theta / half_fov * r
        map_x[sel] = (off + cx + rr * np.cos(phi))[sel]
        map_y[sel] = (cy + rr * np.sin(phi))[sel]
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------- video


def sample_video(path: str | Path, n: int = 12, max_dim: int | None = 1600,
                 around_s: float | None = None) -> list[tuple[float, np.ndarray]]:
    """Grab up to n frames spread across the clip (or within ±2 s of around_s)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaError(f"Could not open video {Path(path).name}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[tuple[float, np.ndarray]] = []
        if count <= 0:
            # Unknown length: read sequentially, keep every 10th frame, then thin out
            idx = 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if idx % 10 == 0:
                    frames.append((idx / fps, resize_max(f, max_dim)))
                idx += 1
            if len(frames) > n:
                keep = np.linspace(0, len(frames) - 1, n).astype(int)
                frames = [frames[i] for i in keep]
            return frames
        if around_s is not None:
            lo = max(0, int((around_s - 2) * fps))
            hi = min(count - 1, int((around_s + 2) * fps))
        else:
            lo, hi = 0, count - 1
        for idx in sorted(set(np.linspace(lo, hi, n).astype(int).tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if ok:
                frames.append((round(idx / fps, 2), resize_max(f, max_dim)))
    finally:
        cap.release()
    if not frames:
        raise MediaError(f"No frames could be decoded from {Path(path).name}")
    return frames


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _quick_features(img: np.ndarray):
    small = resize_max(img, 640)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1500)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def _quick_inliers(fa, fb) -> int:
    kpa, da = fa
    kpb, db = fb
    if da is None or db is None or len(kpa) < 8 or len(kpb) < 8:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(da, db, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 8:
        return 0
    src = np.float32([kpa[m.queryIdx].pt for m in good])
    dst = np.float32([kpb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return int(mask.sum()) if mask is not None else 0


def best_matching_frames(before: list[tuple[float, np.ndarray]], after: list[tuple[float, np.ndarray]],
                         top_k: int = 4) -> dict:
    """Pick the before/after frame pair that best overlaps (most RANSAC inliers among sharp frames)."""
    if not before or not after:
        raise MediaError("Need at least one frame from each input")
    order = sorted(range(len(before)), key=lambda i: sharpness(before[i][1]), reverse=True)[:top_k]
    feats_a = [_quick_features(f) for _, f in after]
    best = (-1, 0, 0)
    for i in order:
        fb = _quick_features(before[i][1])
        for j, fa in enumerate(feats_a):
            score = _quick_inliers(fb, fa)
            if score > best[0]:
                best = (score, i, j)
    score, i, j = best
    return {
        "before_time_s": before[i][0], "before": before[i][1],
        "after_time_s": after[j][0], "after": after[j][1],
        "match_inliers": max(score, 0),
    }


# ---------------------------------------------------------------- 360 equirectangular


def equirect_to_perspective(eq: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0,
                            fov_deg: float = 90.0, out_w: int = 960, out_h: int = 720) -> np.ndarray:
    """Render a pinhole view from an equirectangular panorama."""
    H, W = eq.shape[:2]
    f = 0.5 * out_w / np.tan(np.radians(fov_deg) / 2)
    xs = np.arange(out_w, dtype=np.float32) - (out_w - 1) / 2
    ys = np.arange(out_h, dtype=np.float32) - (out_h - 1) / 2
    x, y = np.meshgrid(xs, ys)
    z = np.full_like(x, f)
    p, t = np.radians(pitch_deg), np.radians(yaw_deg)
    # pitch about x (positive looks up), then yaw about y
    y1 = y * np.cos(p) - z * np.sin(p)
    z1 = y * np.sin(p) + z * np.cos(p)
    x2 = x * np.cos(t) + z1 * np.sin(t)
    z2 = -x * np.sin(t) + z1 * np.cos(t)
    norm = np.sqrt(x2 ** 2 + y1 ** 2 + z2 ** 2)
    lon = np.arctan2(x2, z2)
    lat = np.arcsin(np.clip(-y1 / norm, -1, 1))
    map_x = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32) % W
    map_y = np.clip((0.5 - lat / np.pi) * H, 0, H - 1).astype(np.float32)
    return cv2.remap(eq, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def estimate_yaw_offset(eq_before: np.ndarray, eq_after: np.ndarray) -> tuple[float, float]:
    """Heading difference between two panoramas (degrees, after relative to before) by
    circular phase correlation on the horizon band. Returns (yaw_deg, confidence)."""
    def band(img):
        g = cv2.cvtColor(resize_max(img, 1024), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = cv2.resize(g, (1024, 512), interpolation=cv2.INTER_AREA)
        return np.float32(g[128:384])

    a, b = band(eq_before), band(eq_after)
    (dx, _dy), response = cv2.phaseCorrelate(a, b)
    return float(dx / 1024 * 360.0), float(response)
