"""Optional YOLO-nano labelling via OpenCV DNN (CPU, no PyTorch on the Pi).

Drop an ONNX export of a YOLOv8/YOLO11 nano model at /var/lib/change-detect/models/yolo11n.onnx
(or set YOLO_MODEL). Without it, changes are still found; they just are not labelled.
"""

from __future__ import annotations

import logging
import threading
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

    device = "cpu"

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


# ---------------------------------------------------------------- laptop path (PyTorch / ultralytics)


def detect_accelerator() -> str:
    """cuda > mps > cpu. Only imports torch if it is installed (it is not on the Pi)."""
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def default_weights(device: str) -> str:
    # bigger model when a GPU is present; nano on CPU
    return "yolo11m.pt" if device in ("cuda", "mps") else "yolo11n.pt"


class UltralyticsLabeler:
    """YOLO11 via ultralytics on CUDA/MPS/CPU. Loads in the background so startup is instant."""

    def __init__(self, models_dir: Path, weights: str | None = None, device: str | None = None):
        self.device = device or detect_accelerator()
        self.weights = weights or default_weights(self.device)
        self.model_path = Path(self.weights) if Path(self.weights).is_absolute() else Path(models_dir) / self.weights
        self.model = None
        self.error: str | None = "loading"
        self.imgsz = 1280 if self.device in ("cuda", "mps") else 640
        threading.Thread(target=self._load, name="yolo-load", daemon=True).start()

    def _load(self):
        try:
            from ultralytics import YOLO
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            # downloads the weights into models/ on first run (needs internet once)
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            self.error = None
            logger.info("YOLO %s loaded on %s", self.model_path.name, self.device)
        except Exception as e:
            self.error = f"could not load {self.weights} ({e})"
            logger.warning("YOLO disabled: %s", self.error)

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def detect(self, img: np.ndarray) -> list[Detection]:
        if self.model is None:
            return []
        res = self.model.predict(img, device=self.device, imgsz=self.imgsz, conf=0.35, verbose=False)[0]
        dets = []
        names = res.names
        for (x0, y0, x1, y1), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            name = names.get(int(k), str(int(k))) if isinstance(names, dict) else COCO[int(k)]
            dets.append(Detection(group=GROUPS.get(name, name), cls=name, conf=round(float(c), 2),
                                  box=(int(x0), int(y0), int(x1 - x0), int(y1 - y0))))
        return dets


def make_labeler(backend: str, onnx_path: Path | None, models_dir: Path, weights: str | None = None):
    """auto: ultralytics if installed (laptop), else the OpenCV-DNN ONNX model (Pi), else off."""
    backend = (backend or "auto").lower()
    if backend == "off":
        lab = Labeler(None)
        lab.error = "disabled (YOLO_BACKEND=off)"
        return lab
    if backend in ("auto", "ultralytics"):
        try:
            import ultralytics  # noqa: F401
            return UltralyticsLabeler(models_dir, weights)
        except Exception as e:
            if backend == "ultralytics":
                lab = Labeler(None)
                lab.error = f"ultralytics not installed ({e})"
                return lab
    return Labeler(onnx_path)
