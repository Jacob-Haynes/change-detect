"""Optional YOLO-nano labelling via OpenCV DNN (CPU, no PyTorch on the Pi).

Drop an ONNX export of a YOLOv8/YOLO11 nano model at /var/lib/change-detect/models/yolo11n.onnx
(or set YOLO_MODEL). Without it, changes are still found; they just are not labelled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]

GROUPS = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle",
          "bicycle": "vehicle", "train": "vehicle", "boat": "boat", "airplane": "aircraft",
          "person": "person", "backpack": "kit", "suitcase": "kit", "handbag": "kit",
          "dog": "animal", "horse": "animal", "sheep": "animal", "cow": "animal"}


@dataclass
class Detection:
    group: str
    cls: str
    conf: float
    box: tuple[int, int, int, int]


def decode_yolo(output: np.ndarray, img_w: int, img_h: int, in_size: int = 640,
                conf_thr: float = 0.35, nms_thr: float = 0.45) -> list[Detection]:
    """Decode a YOLOv8/11 ONNX output tensor of shape (1, 4+nc, N) or (1, N, 4+nc)."""
    out = np.squeeze(output)
    if out.ndim != 2:
        return []
    nch = 4 + len(COCO)
    if out.shape[0] == nch or (out.shape[1] != nch and out.shape[0] < out.shape[1]):
        out = out.T  # -> (N, 4+nc)
    if out.shape[1] <= 4:
        return []
    boxes_xywh, scores = out[:, :4], out[:, 4:]
    classes = scores.argmax(1)
    confs = scores[np.arange(len(scores)), classes]
    keep = confs >= conf_thr
    if not keep.any():
        return []
    sx, sy = img_w / in_size, img_h / in_size
    rects, cf, cl = [], [], []
    for (cx, cy, w, h), c, k in zip(boxes_xywh[keep], confs[keep], classes[keep]):
        rects.append([int((cx - w / 2) * sx), int((cy - h / 2) * sy), int(w * sx), int(h * sy)])
        cf.append(float(c))
        cl.append(int(k))
    idx = cv2.dnn.NMSBoxes(rects, cf, conf_thr, nms_thr)
    dets = []
    for i in np.array(idx).reshape(-1):
        name = COCO[cl[i]] if cl[i] < len(COCO) else str(cl[i])
        dets.append(Detection(group=GROUPS.get(name, name), cls=name, conf=round(cf[i], 2), box=tuple(rects[i])))
    return dets


class Labeler:
    def __init__(self, model_path: Path | None):
        self.model_path = model_path
        self.net = None
        self.error: str | None = None
        if model_path and Path(model_path).exists():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(model_path))
                logger.info("YOLO model loaded: %s", model_path)
            except Exception as e:  # bad/unsupported export
                self.error = f"could not load {Path(model_path).name}: {e}"
                logger.warning("YOLO disabled: %s", self.error)
        else:
            self.error = "no model file"

    @property
    def enabled(self) -> bool:
        return self.net is not None

    def detect(self, img: np.ndarray) -> list[Detection]:
        if self.net is None:
            return []
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        return decode_yolo(self.net.forward(), w, h)


def _overlaps(box, det_box, pad: float = 0.25) -> bool:
    x, y, w, h = box
    px, py = w * pad, h * pad
    x0, y0, x1, y1 = x - px, y - py, x + w + px, y + h + py
    dx, dy, dw, dh = det_box
    ix = max(0, min(x1, dx + dw) - max(x0, dx))
    iy = max(0, min(y1, dy + dh) - max(y0, dy))
    inter = ix * iy
    return inter > 0.3 * max(1, dw * dh) or inter > 0.3 * (w * h)


def describe(box, before_dets: list[Detection], after_dets: list[Detection]) -> tuple[list[str], str | None]:
    """Turn detections around a change box into labels like 'vehicle appeared'."""
    b = [d for d in before_dets if _overlaps(box, d.box)]
    a = [d for d in after_dets if _overlaps(box, d.box)]
    groups = sorted({d.group for d in b} | {d.group for d in a})
    phrases = []
    for g in groups:
        nb = sum(1 for d in b if d.group == g)
        na = sum(1 for d in a if d.group == g)
        if na > nb:
            phrases.append(f"{g} appeared")
        elif nb > na:
            phrases.append(f"{g} gone")
        else:
            phrases.append(f"{g} moved/changed")
    labels = [f"before:{d.cls} {d.conf:.2f}" for d in b] + [f"after:{d.cls} {d.conf:.2f}" for d in a]
    return labels, ", ".join(phrases) or None
