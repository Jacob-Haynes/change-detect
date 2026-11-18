"""Fetch every model weight change-detect needs, so it runs offline on the hotspot.

    python -m changedetect.prefetch [--data-dir DIR] [--export-onnx] [--from DIR] [--strict]

- Mac/desktop (ultralytics installed): downloads the YOLO weights the default profile uses
  (yolo11m.pt on MPS/CUDA, yolo11n.pt on CPU) plus yolo11n.pt as the fallback, into <data>/models.
  `--export-onnx` also writes yolo11n.onnx there for copying to the Pi.
- Pi (OpenCV DNN, no ultralytics): makes sure <data>/models/yolo11n.onnx exists, by copying it
  from `--from DIR` or downloading YOLO_ONNX_URL. Otherwise it says how to get one.

The local VLM (Ollama) is pulled by scripts/run-laptop.sh --setup-only, not here. YOLO is optional,
so a missing model is a warning (exit 0) unless --strict.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from changedetect.config import Settings
from changedetect.labeler import default_weights, detect_accelerator

ONNX_NAME = "yolo11n.onnx"


def say(msg: str) -> None:
    print(f"[prefetch] {msg}", flush=True)


def wanted_pt_weights(device: str) -> list[str]:
    names = [os.environ.get("YOLO_WEIGHTS") or default_weights(device), "yolo11n.pt"]
    return list(dict.fromkeys(n for n in names if n.endswith(".pt")))


def fetch_pt(models_dir: Path, name: str) -> bool:
    dest = models_dir / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        say(f"ok      {dest} ({dest.stat().st_size // 1_000_000} MB, already present)")
        return True
    from ultralytics.utils.downloads import attempt_download_asset
    say(f"fetch   {name} -> {models_dir}")
    try:
        got = Path(attempt_download_asset(dest))
    except Exception as e:
        say(f"FAILED  {name}: {e}")
        return False
    if got.exists() and got.resolve() != dest.resolve():
        shutil.copy2(got, dest)  # ultralytics found it in its own weights dir
    if dest.exists() and dest.stat().st_size > 1_000_000:
        say(f"ok      {dest} ({dest.stat().st_size // 1_000_000} MB)")
        return True
    say(f"FAILED  {name}: no file after download (offline?)")
    return False


def export_onnx(models_dir: Path) -> bool:
    dest = models_dir / ONNX_NAME
    if dest.exists():
        say(f"ok      {dest} (already present)")
        return True
    if not fetch_pt(models_dir, "yolo11n.pt"):
        return False
    from ultralytics import YOLO
    say(f"export  {ONNX_NAME} (opset 12, 640 px) for the Pi's OpenCV DNN")
    try:
        out = Path(YOLO(str(models_dir / "yolo11n.pt")).export(format="onnx", imgsz=640, opset=12,
                                                               simplify=False, verbose=False))
    except Exception as e:
        say(f"FAILED  ONNX export: {e}")
        return False
    if out.resolve() != dest.resolve():
        shutil.move(str(out), dest)
    say(f"ok      {dest}: copy it to the Pi (for example into the repo's models/ before deploy/install.sh)")
    return True


def ensure_onnx(dest: Path, search: list[Path]) -> bool:
    if dest.exists():
        say(f"ok      {dest} (already present)")
        return True
    for d in search:
        cand = d / dest.name
        if cand.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, dest)
            say(f"ok      {dest} (copied from {cand})")
            return True
    url = os.environ.get("YOLO_ONNX_URL")
    if url:
        say(f"fetch   {url} -> {dest}")
        tmp = dest.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=30) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if tmp.stat().st_size < 1_000_000:
                raise OSError("file too small to be a YOLO model")
            tmp.replace(dest)
            say(f"ok      {dest}")
            return True
        except Exception as e:
            tmp.unlink(missing_ok=True)
            say(f"FAILED  {url}: {e}")
            return False
    say(f"MISSING {dest}: YOLO labels will be off. On a Mac run "
        "'.venv-laptop/bin/python -m changedetect.prefetch --export-onnx' and copy the file to the Pi, "
        "or set YOLO_ONNX_URL.")
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m changedetect.prefetch", description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--export-onnx", action="store_true", help="also write yolo11n.onnx for the Pi (needs ultralytics)")
    p.add_argument("--from", dest="src", action="append", default=[], help="directory to copy a ready .onnx from")
    p.add_argument("--strict", action="store_true", help="exit 1 if any model is missing")
    args = p.parse_args(argv)

    settings = Settings.from_env(args.data_dir)
    models_dir = settings.data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    backend = (settings.yolo_backend or "auto").lower()
    ok = True
    if backend == "off":
        say("YOLO_BACKEND=off: no YOLO weights needed")
    else:
        try:
            import ultralytics  # noqa: F401
            have_ul = True
        except Exception:
            have_ul = False
        if have_ul and backend in ("auto", "ultralytics"):
            device = detect_accelerator()
            say(f"accelerator {device}")
            for name in wanted_pt_weights(device):
                ok &= fetch_pt(models_dir, name)
            if args.export_onnx:
                ok &= export_onnx(models_dir)
        elif backend == "ultralytics":
            say("YOLO_BACKEND=ultralytics but ultralytics is not installed (pip install -r requirements-desktop.txt)")
            ok = False
        else:
            ok &= ensure_onnx(settings.yolo_model or models_dir / ONNX_NAME, [Path(s) for s in args.src])
    say("all models present" if ok else "some models are missing (see above)")
    return 1 if (args.strict and not ok) else 0


if __name__ == "__main__":
    sys.exit(main())
