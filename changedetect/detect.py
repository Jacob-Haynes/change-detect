"""Illumination normalisation, differencing and change-blob extraction (pure OpenCV/NumPy)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_CHANGES = 25
NOISY_BLOB_COUNT = 40


@dataclass
class Change:
    box: tuple[int, int, int, int]  # x, y, w, h in BEFORE frame
    area: int
    score: float
    kind: str
    n: int = 0
    labels: list[str] = field(default_factory=list)
    label_text: str | None = None
    ai_suggestion: str | None = None

    def to_dict(self) -> dict:
        x, y, w, h = self.box
        return {
            "n": self.n, "box": [int(x), int(y), int(w), int(h)], "area": int(self.area),
            "score": round(float(self.score), 2), "kind": self.kind, "labels": self.labels,
            "label_text": self.label_text, "ai_suggestion": self.ai_suggestion,
        }


@dataclass
class DiffResult:
    score_map: np.ndarray  # float32, 0..~2, 1.0 == threshold at default sensitivity
    mask: np.ndarray       # bool change mask
    valid: np.ndarray      # bool comparable area
    threshold: float
    changes: list[Change]
    warnings: list[str]
    after_normalised: np.ndarray


def normalise_illumination(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Match AFTER's LAB channel statistics to BEFORE inside the overlap (global light/white balance)."""
    lb = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    la = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
    if valid.sum() < 100:
        return after
    out = la.copy()
    for c in range(3):
        mb, sb = lb[..., c][valid].mean(), lb[..., c][valid].std() + 1e-3
        ma, sa = la[..., c][valid].mean(), la[..., c][valid].std() + 1e-3
        gain = np.clip(sb / sa, 0.5, 2.0) if c == 0 else np.clip(sb / sa, 0.7, 1.4)
        out[..., c] = (la[..., c] - ma) * gain + mb
    out = np.clip(out, 0, 255).astype(np.uint8)
    res = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    res[~valid] = 0
    return res


def _ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)
    mu_a, mu_b = blur(a), blur(b)
    saa = blur(a * a) - mu_a ** 2
    sbb = blur(b * b) - mu_b ** 2
    sab = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)
    return num / den


def _shift_min_absdiff(a: np.ndarray, b: np.ndarray, r: int) -> np.ndarray:
    """|a-b| minimised over small shifts of b: tolerates 1-2 px of residual misregistration."""
    best = None
    pad = cv2.copyMakeBorder(b, r, r, r, r, cv2.BORDER_REPLICATE)
    h, w = a.shape[:2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = pad[r + dy:r + dy + h, r + dx:r + dx + w]
            d = cv2.absdiff(a, shifted)
            if d.ndim == 3:
                d = np.sqrt((d.astype(np.float32) ** 2).sum(axis=2))
            else:
                d = d.astype(np.float32)
            best = d if best is None else np.minimum(best, d)
    return best


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 150)
    return float((edges > 0).mean())


def classify(before: np.ndarray, after: np.ndarray, box) -> str:
    x, y, w, h = box
    gb = cv2.cvtColor(before[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    ga = cv2.cvtColor(after[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    eb, ea = _edge_density(gb) + 0.005, _edge_density(ga) + 0.005
    if ea / eb > 1.6:
        return "new object / feature"
    if eb / ea > 1.6:
        return "object / feature removed"
    lb = cv2.cvtColor(before[y:y + h, x:x + w], cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
    la = cv2.cvtColor(after[y:y + h, x:x + w], cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
    if np.linalg.norm(la[1:] - lb[1:]) > 6 or abs(la[0] - lb[0]) > 18:
        return "surface / colour change"
    return "object moved or changed"


def detect_changes(before: np.ndarray, after_aligned: np.ndarray, valid: np.ndarray,
                   sensitivity: float = 0.5, min_area: int = 300) -> DiffResult:
    """Return change blobs between BEFORE and the aligned AFTER image.

    sensitivity 0..1 (higher = more changes); min_area in processing-resolution pixels.
    """
    h, w = before.shape[:2]
    scale = max(h, w) / 1280.0
    warnings: list[str] = []
    # min_area is specified at ~1600 px width; scale it to the real resolution
    min_area = int(min_area * (max(h, w) / 1600.0) ** 2) if max(h, w) > 1600 else min_area
    # Ignore the ragged edge of the warped image
    erode_px = max(3, int(round(6 * scale)))
    valid_e = cv2.erode(valid.astype(np.uint8), np.ones((erode_px * 2 + 1,) * 2, np.uint8)) > 0

    norm_after = normalise_illumination(before, after_aligned, valid_e)
    k = max(3, int(round(3 * scale)) | 1)
    bb = cv2.GaussianBlur(before, (k, k), 0)
    ab = cv2.GaussianBlur(norm_after, (k, k), 0)
    lab_b = cv2.cvtColor(bb, cv2.COLOR_BGR2LAB)
    lab_a = cv2.cvtColor(ab, cv2.COLOR_BGR2LAB)

    r = min(3, max(1, int(round(2 * scale))))
    d_l = _shift_min_absdiff(lab_b[..., 0], lab_a[..., 0], r)
    d_c = _shift_min_absdiff(lab_b[..., 1:], lab_a[..., 1:], r)
    ssim_d = 1.0 - _ssim_map(cv2.GaussianBlur(lab_b[..., 0], (0, 0), 1.0),
                             cv2.GaussianBlur(lab_a[..., 0], (0, 0), 1.0), sigma=2.0 * scale)
    # Structural difference only counts where there is also an intensity/colour difference:
    # SSIM alone flags texture noise.
    support = np.maximum(d_l / 25.0, d_c / 15.0)
    score = np.maximum.reduce([d_l / 45.0, d_c / 22.0, (ssim_d / 0.7) * np.clip(support, 0, 1)])
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 1.5 * scale)
    score[~valid_e] = 0

    sens = float(np.clip(sensitivity, 0.0, 1.0))
    threshold = 1.5 - sens  # 0.5 sensitivity -> 1.0
    vals = score[valid_e]
    if vals.size:
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-6
        adaptive = med + 8.0 * 1.4826 * mad
        if adaptive > threshold:
            warnings.append("high background difference (lighting, season, motion blur or parallax); "
                            "threshold raised automatically")
            threshold = min(adaptive, threshold * 1.6)

    mask = (score > threshold).astype(np.uint8)
    o = max(3, int(round(3 * scale)) | 1)
    c = max(5, int(round(9 * scale)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (o, o)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))
    mask &= valid_e.astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        blob_score = float(score[labels == i].mean())
        blobs.append((area * blob_score, Change(box=(int(x), int(y), int(bw), int(bh)), area=int(area),
                                                score=blob_score, kind="")))
    if len(blobs) > NOISY_BLOB_COUNT:
        warnings.append(f"{len(blobs)} candidate changes: likely misalignment, parallax or a very "
                        "different scene; treat results with caution")
    blobs.sort(key=lambda t: t[0], reverse=True)
    changes = [b for _, b in blobs[:MAX_CHANGES]]
    for ch in changes:
        ch.kind = classify(before, norm_after, ch.box)
    final_mask = np.zeros((h, w), bool)
    for i in range(1, n):
        if stats[i][4] >= min_area:
            final_mask |= labels == i
    return DiffResult(score_map=score, mask=final_mask, valid=valid_e, threshold=float(threshold),
                      changes=changes, warnings=warnings, after_normalised=norm_after)
