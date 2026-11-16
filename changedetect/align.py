"""Feature matching + RANSAC homography, with an honest quality grade.

The homography maps the AFTER image into the BEFORE image's pixel frame. A homography
is exact only for a flat scene (or a camera that only rotates), so tall objects seen
from a different position will always leave some parallax residue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

MIN_INLIERS_FAIL = 15
MIN_INLIERS_GOOD = 40
MIN_RATIO_FAIL = 0.12
MIN_RATIO_GOOD = 0.30
MIN_OVERLAP_FAIL = 0.30
MIN_OVERLAP_GOOD = 0.60
MAX_REPROJ_GOOD = 2.5
MIN_SPREAD_FAIL = 0.04
MIN_SPREAD_GOOD = 0.15


@dataclass
class Alignment:
    ok: bool = False
    grade: str = "failed"  # good | marginal | failed
    reasons: list[str] = field(default_factory=list)
    method: str = ""
    keypoints_before: int = 0
    keypoints_after: int = 0
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reproj_error_px: float | None = None
    overlap: float = 0.0
    inlier_spread: float = 0.0
    scale: float | None = None
    rotation_deg: float | None = None
    H: list | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("H", None)
        return d

    @property
    def homography(self) -> np.ndarray | None:
        return None if self.H is None else np.array(self.H, dtype=np.float64)


def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # CLAHE makes features more repeatable under different light
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def available_methods() -> list[str]:
    """AKAZE where this OpenCV build has it (removed from OpenCV 5 core), then ORB, then SIFT."""
    methods = ["akaze"] if hasattr(cv2, "AKAZE_create") else []
    methods.append("orb")
    if hasattr(cv2, "SIFT_create"):
        methods.append("sift")
    return methods


def _detect(gray: np.ndarray, method: str):
    if method == "akaze":
        det = cv2.AKAZE_create(threshold=0.0008)
    elif method == "sift":
        det = cv2.SIFT_create(nfeatures=4000)
    else:
        det = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    return det.detectAndCompute(gray, None)


def _convex(quad: np.ndarray) -> bool:
    signs = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def align(before: np.ndarray, after: np.ndarray) -> Alignment:
    gb, ga = _gray(before), _gray(after)
    h, w = gb.shape[:2]
    best: Alignment | None = None
    for method in available_methods():
        res = _align_with(gb, ga, method, w, h)
        if best is None or res.inliers > best.inliers:
            best = res
        if res.grade == "good":
            break
    return best


def _align_with(gb, ga, method, w, h) -> Alignment:
    res = Alignment(method=method)
    kb, db = _detect(gb, method)
    ka, da = _detect(ga, method)
    res.keypoints_before, res.keypoints_after = len(kb), len(ka)
    if db is None or da is None or len(kb) < 10 or len(ka) < 10:
        res.reasons.append("too few features (featureless scene, blur or very dark)")
        return res
    matcher = cv2.BFMatcher(cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(da, db, k=2):
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
            good.append(pair[0])
    res.matches = len(good)
    if len(good) < MIN_INLIERS_FAIL:
        res.reasons.append(f"only {len(good)} feature matches; images may not show the same area")
        return res
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.995)
    if H is None or mask is None:
        res.reasons.append("RANSAC found no consistent transform")
        return res
    inl = mask.ravel().astype(bool)
    res.inliers = int(inl.sum())
    res.inlier_ratio = round(res.inliers / max(len(good), 1), 3)
    proj = cv2.perspectiveTransform(src[inl], H)
    err = np.linalg.norm(proj - dst[inl], axis=2).ravel()
    res.reproj_error_px = round(float(np.median(err)), 2) if err.size else None

    # Geometry sanity: warped AFTER frame must be a plausible convex quad of similar size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    area_ratio = abs(cv2.contourArea(quad)) / float(w * h)
    A = H[:2, :2] / H[2, 2]
    res.scale = round(float(np.sqrt(abs(np.linalg.det(A)))), 3)
    res.rotation_deg = round(float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), 1)
    sane = _convex(quad) and 0.2 < area_ratio < 5.0

    # Overlap: share of BEFORE frame covered by the warped AFTER frame
    m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (w, h))
    res.overlap = round(float((m > 0).mean()), 3)

    # Spread: are inliers spread across the frame, or clustered in one corner?
    pts = dst[inl].reshape(-1, 2)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts.astype(np.float32))
        res.inlier_spread = round(float(cv2.contourArea(hull)) / float(w * h), 3)
    res.H = H.tolist()

    fail, warn = [], []
    if not sane:
        fail.append("transform is geometrically implausible (flipped or extreme warp)")
    if res.inliers < MIN_INLIERS_FAIL:
        fail.append(f"only {res.inliers} inliers")
    elif res.inliers < MIN_INLIERS_GOOD:
        warn.append(f"low inlier count ({res.inliers})")
    if res.inlier_ratio < MIN_RATIO_FAIL:
        fail.append(f"inlier ratio {res.inlier_ratio:.2f} too low")
    elif res.inlier_ratio < MIN_RATIO_GOOD:
        warn.append(f"inlier ratio {res.inlier_ratio:.2f} is low")
    if res.overlap < MIN_OVERLAP_FAIL:
        fail.append(f"images overlap only {res.overlap:.0%}")
    elif res.overlap < MIN_OVERLAP_GOOD:
        warn.append(f"partial overlap ({res.overlap:.0%})")
    if res.inlier_spread < MIN_SPREAD_FAIL:
        fail.append("matches clustered in a tiny area")
    elif res.inlier_spread < MIN_SPREAD_GOOD:
        warn.append("matches clustered; edges of frame less reliable")
    if res.reproj_error_px is not None and res.reproj_error_px > MAX_REPROJ_GOOD:
        warn.append(f"residual error {res.reproj_error_px:.1f}px (parallax or lens distortion)")

    if fail:
        res.grade, res.ok, res.reasons = "failed", False, fail + warn
    elif warn:
        res.grade, res.ok, res.reasons = "marginal", True, warn
    else:
        res.grade, res.ok = "good", True
    return res


def warp_after(after: np.ndarray, H: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Warp AFTER into BEFORE's frame; returns (warped image, valid mask)."""
    h, w = shape[:2]
    warped = cv2.warpPerspective(after, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(np.full(after.shape[:2], 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
    return warped, valid > 0
