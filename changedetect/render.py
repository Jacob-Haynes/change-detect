"""Annotated images and the exportable PNG/PDF report."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BOX_COLOUR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_boxes(img: np.ndarray, changes, colour=BOX_COLOUR) -> np.ndarray:
    out = img.copy()
    t = max(2, round(max(img.shape[:2]) / 600))
    for ch in changes:
        x, y, w, h = ch.box
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, t)
        label = str(ch.n)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5 * t, t)
        ty = y - 4 if y - th - 8 > 0 else y + h + th + 4
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 6, ty + 2), colour, -1)
        cv2.putText(out, label, (x + 3, ty - 1), FONT, 0.5 * t, (255, 255, 255), t, cv2.LINE_AA)
    return out


def heatmap(before: np.ndarray, score: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    norm = np.clip(score / max(threshold * 1.8, 1e-3), 0, 1)
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (0.15 + 0.7 * norm)[..., None]
    base = cv2.cvtColor(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = (base * (1 - alpha) + colour * alpha).astype(np.uint8)
    out[~valid] = (out[~valid] * 0.35).astype(np.uint8)
    return out


def side_by_side(left: np.ndarray, right: np.ndarray, labels=("BEFORE", "AFTER (aligned)")) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    gap = np.full((h, 12, 3), 255, np.uint8)
    out = np.hstack([pad(left), gap, pad(right)])
    s = max(0.6, h / 900)
    for x, text in ((10, labels[0]), (left.shape[1] + 22, labels[1])):
        cv2.putText(out, text, (x, int(32 * s)), FONT, s, (0, 0, 0), int(4 * s), cv2.LINE_AA)
        cv2.putText(out, text, (x, int(32 * s)), FONT, s, (255, 255, 255), max(1, int(1.5 * s)), cv2.LINE_AA)
    return out


def crop(img: np.ndarray, box, pad: float = 0.4, min_size: int = 64) -> np.ndarray:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h, min_size) * (0.5 + pad)
    H, W = img.shape[:2]
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
    return img[y0:y1, x0:x1].copy()


def write_jpeg(path: Path, img: np.ndarray, quality: int = 88) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _text_block(lines: list[tuple[str, float, tuple]], width: int) -> np.ndarray:
    line_h = [int(34 * s) + 8 for _, s, _ in lines]
    block = np.full((sum(line_h) + 20, width, 3), 255, np.uint8)
    y = 10
    for (text, s, colour), lh in zip(lines, line_h):
        y += lh
        cv2.putText(block, text[:140], (20, y - 8), FONT, s, colour, max(1, int(2 * s)), cv2.LINE_AA)
    return block


def build_report(result: dict, view_images: dict[str, dict[str, np.ndarray]], out_dir: Path) -> dict:
    """Compose a single long PNG (and a PDF of the same pages) summarising the analysis."""
    W = 1800
    pages: list[np.ndarray] = []
    head = [
        (f"CHANGE DETECTION REPORT  {result['id']}", 1.2, (0, 0, 0)),
        (f"Generated {result['created']}   Mode: {result['mode']}", 0.7, (60, 60, 60)),
        (f"Before: {result['inputs']['before']['name']}  ({result['inputs']['before'].get('time') or 'no time'})", 0.7, (60, 60, 60)),
        (f"After:  {result['inputs']['after']['name']}  ({result['inputs']['after'].get('time') or 'no time'})", 0.7, (60, 60, 60)),
    ]
    gps = result.get("gps")
    head.append((f"Position: {gps['lat']:.6f}, {gps['lon']:.6f} ({gps['source']})" if gps else "Position: none in images",
                 0.7, (60, 60, 60)))
    head.append((result["summary"], 0.8, (0, 0, 160)))
    head.append(("TRAINING / DEMO OUTPUT - machine-generated, verify every change by eye.", 0.6, (0, 0, 200)))
    blocks = [_text_block(head, W)]

    for view in result["views"]:
        al = view["alignment"]
        lines = [(f"View: {view['name']}   Alignment: {al['grade'].upper()} ({al['method']}, {al['inliers']} inliers, "
                  f"ratio {al['inlier_ratio']:.2f}, overlap {al['overlap']:.0%}, error {al['reproj_error_px']} px)",
                  0.7, (0, 0, 0))]
        for r in al.get("reasons", []) + view.get("warnings", []):
            lines.append((f"  ! {r}", 0.6, (0, 0, 180)))
        blocks.append(_text_block(lines, W))
        imgs = view_images.get(view["name"], {})
        if "side" in imgs:
            side = imgs["side"]
            blocks.append(cv2.resize(side, (W, int(side.shape[0] * W / side.shape[1]))))
        if "heat" in imgs and view["changes"]:
            hm = imgs["heat"]
            hw = W // 2
            blocks.append(cv2.copyMakeBorder(cv2.resize(hm, (hw, int(hm.shape[0] * hw / hm.shape[1]))),
                                             0, 0, 0, W - hw, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        for ch in view["changes"]:
            cb, ca = imgs.get(f"crop{ch['n']}_b"), imgs.get(f"crop{ch['n']}_a")
            row_h = 260
            row = np.full((row_h, W, 3), 255, np.uint8)
            x = 20
            for c in (cb, ca):
                if c is None or c.size == 0:
                    continue
                r = cv2.resize(c, (int(c.shape[1] * (row_h - 20) / c.shape[0]), row_h - 20))
                r = r[:, : min(r.shape[1], 400)]
                row[10:10 + r.shape[0], x:x + r.shape[1]] = r
                x += r.shape[1] + 15
            txt = [f"#{ch['n']}  {ch['kind']}", f"area {ch['area']} px  score {ch['score']}"]
            if ch.get("label_text"):
                txt.append(f"YOLO: {ch['label_text']}")
            if ch.get("ai_suggestion"):
                txt.append(f"AI suggestion: {ch['ai_suggestion']}")
            for i, t in enumerate(txt):
                cv2.putText(row, t[:80], (x + 20, 50 + i * 42), FONT, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
            blocks.append(row)

    # Paginate at ~A4 aspect for the PDF; the PNG is the whole thing
    full = np.vstack(blocks)
    png_path = out_dir / "report.png"
    cv2.imwrite(str(png_path), full)
    page_h = int(W * 1.414)
    for y in range(0, full.shape[0], page_h):
        page = full[y:y + page_h]
        if page.shape[0] < page_h:
            page = cv2.copyMakeBorder(page, 0, page_h - page.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        pages.append(page)
    pil = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in pages]
    pdf_path = out_dir / "report.pdf"
    pil[0].save(pdf_path, "PDF", resolution=150.0, save_all=True, append_images=pil[1:])
    return {"png": png_path.name, "pdf": pdf_path.name}
