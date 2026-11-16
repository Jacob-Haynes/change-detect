# change-detect

Compare **before** and **after** imagery of the same area and highlight what changed. This is the
classic recce/ISR task: a vehicle that moved, a new hide, disturbed ground. It runs entirely on a
Raspberry Pi 5 CPU (OpenCV, no AI accelerator) with a web UI on **port 8130**.

> Training and demonstration tool. Every change it reports is a machine guess: confirm each one by eye.
> It is defensive and observational only.

## What it does

1. **Inputs** (web UI): upload two stills or two videos, or pick a bundled sample. Stills can be
   JPEG, PNG or iPhone HEIC. Videos can be MP4/MOV, including DJI O3/O4 onboard recordings
   (H.264/H.265, 4K, usually no GPS). For videos it samples frames from both clips and picks the
   sharpest pair that overlaps best; you can give an approximate time in each clip. EXIF GPS,
   capture time and camera are read where present.
   **Folder / SD card:** drag a DCIM folder straight from the card. One folder is split into two
   passes at the largest gap in capture time. Or drop each pass in its own box. Matching shots are
   paired by GPS (within 40 m), by position in the sequence and by feature matching, and each pair
   is analysed. Sidecar files (`.LRV`, `.THM`, `._*`) are skipped.
2. **360 stills** (equirectangular, 2:1, or tagged `GPano:ProjectionType=equirectangular`): it
   estimates the heading difference between the two panoramas, then compares four matching
   perspective views (front/right/back/left).
3. **Align**: AKAZE features (ORB/SIFT on OpenCV 5, which dropped AKAZE from core), ratio-test
   matching, RANSAC homography. The alignment is graded **good / marginal / failed** from the inlier
   count, inlier ratio, overlap, how spread out the matches are, residual error and a geometry
   check. **If alignment fails, no changes are called.** The UI says why.
4. **Compare**: LAB colour-statistics matching (illumination normalisation), a blur, then a combined
   score from shift-tolerant abs-diff on L and on colour, plus SSIM. Morphology turns that into change
   blobs, filtered by minimum area. If the background difference is high, the threshold is raised
   and a warning is shown.
5. **Label** (optional): YOLO nano (ONNX, run through OpenCV DNN on the CPU) runs on both images.
   It labels a change "vehicle appeared", "person gone" and so on.
6. **Outputs**: side-by-side, swipe slider, heatmap overlay, numbered change boxes (click to toggle
   before/after), per-change before/after crops, and a **PNG + PDF report** saved to
   `/var/lib/change-detect/reports/<id>/` (with the original files and `result.json`).
7. **Optional AI suggestion:** sends the two aligned crops of each change to a vision model and
   shows a one-line description labelled "AI suggestion (source)". `VLM_BACKEND=auto` tries a local
   OpenAI-compatible endpoint first (Ollama on the HQ Mac, `qwen3-vl:8b`, offline), then Gemini
   if `GEMINI_API_KEY` is set, otherwise off. `openai_compat`, `gemini` or `off` forces one backend.
8. **Optional ATAK**: pushes a CoT "change detected" marker (`a-u-G`, unknown ground) with the
   summary, using EXIF GPS or a manually entered lat/lon. **Mesh**: sends a short summary text over
   the Meshtastic mesh via the blufor-tracker bridge.

The header always shows which modes are active: `SIMULATE`, profile and accelerator
(`PI · CPU · 1600px`, `LAPTOP · MPS · full-res`), the AI suggestion source (local VLM, Gemini
cloud, or off), YOLO on/off and its device, TAK connection, and mesh state.

Every 10 s it publishes a retained heartbeat to `isrkit/heartbeat/change-detect`, so the Pi
launcher can link to it wherever it runs. The payload is
`{"tool":"change-detect","host":"isr-hq","url":"http://isr-hq.local:8130/","mode":"mps","status":"ok","ts":…}`.
- `mode` is `cpu`, `mps`, `cuda` or `simulate`.
- `status` is `ok`, or `degraded` when ATAK markers are stuck in the queue or a mesh send failed.
- On a clean exit (SIGTERM) it publishes a retained `stopped`. If the process dies uncleanly, the
  MQTT last-will publishes `error`.

## Install on the Pi

```bash
git clone <this repo> && cd change-detect
sudo bash deploy/install.sh
sudo pi-profiles change-detect          # standalone
sudo pi-profiles bft-change-detect      # with OpenTAKServer + mesh bridge, for ATAK markers and mesh texts
```

- Code: `/opt/change-detect` (venv in `.venv`). Data and reports: `/var/lib/change-detect`.
  Config: `/etc/change-detect.env` (created from `deploy/change-detect.env.example`, root-only).
- The service runs as the `changedetect` system user and is **not** enabled at boot: pi-profiles
  starts it.
- Logs: `journalctl -u change-detect -f`.

### Configuration (`/etc/change-detect.env`)

| Variable | Default | Purpose |
|---|---|---|
| `TAK_HOST` / `TAK_PORT` | `127.0.0.1` / `8087` | CoT TCP output. If the TAK server is down, markers queue and retry with backoff. |
| `MQTT_HOST` / `MQTT_PORT` / `MQTT_USER` / `MQTT_PASS` | `127.0.0.1` / `1883` / `guest` / `guest` | Publishes to `meshbus/tx`. It never opens the radio itself. |
| `VLM_BACKEND` | `auto` | `auto` (local VLM → Gemini → off), `openai_compat`, `gemini` or `off`. |
| `VLM_URL` / `VLM_MODEL` | `http://localhost:11434` / `qwen3-vl:8b` | **Bare** Ollama base URL, with no path (the tool appends `/v1/chat/completions`). On the Pi, set `http://isr-hq.local:11434`. |
| `GEMINI_API_KEY` | empty | Empty means no cloud. Set it to allow Gemini suggestions. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any current Gemini model that accepts images. |
| `YOLO_BACKEND` | `auto` | `auto` uses ultralytics (PyTorch) if installed (laptop), else ONNX via OpenCV (Pi). `off` disables YOLO. |
| `YOLO_MODEL` | `/var/lib/change-detect/models/yolo11n.onnx` | ONNX model for the Pi path. |
| `YOLO_WEIGHTS` | `yolo11m.pt` on MPS/CUDA, `yolo11n.pt` on CPU | Ultralytics weights for the laptop path. |
| `CHANGE_DETECT_PROFILE` | `pi` | `pi` downscales stills to 1600 px. `laptop` keeps full resolution. |
| `HEARTBEAT` / `PUBLIC_URL` | `1` / auto | Heartbeat on/off, and the URL it advertises. |
| `CHANGE_DETECT_ARGS` | empty | e.g. `--simulate` or `--no-cot`. |

### Optional: YOLO nano labels

YOLO is not bundled: the weights come under Ultralytics' AGPL-3.0 licence, and the Pi needs no
PyTorch. On a laptop with internet access:

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=onnx opset=12 imgsz=640
scp yolo11n.onnx pi@<pi>:/tmp/ && ssh pi@<pi> 'sudo install -o changedetect -m 644 /tmp/yolo11n.onnx /var/lib/change-detect/models/'
sudo systemctl restart change-detect
```

COCO classes are trained on ground-level photos. Expect good results on GoPro and phone shots,
and poor results on top-down drone imagery from height.

### Optional: Gemini cloud mode

> **Cloud mode sends image crops to Google.** Use demo footage only. Never use real operational
> imagery. With no key, or with no internet, everything else works offline.

## Running on a laptop

The Pi `isr-pi.local` (hotspot 192.168.4.1) stays the hub: hotspot, OpenTAKServer, mesh bridge,
MQTT broker and launcher. change-detect runs on the **HQ Mac** `isr-hq.local` (M5 Pro, which also
runs Ollama) as the analysis station. The Mac joins the Pi's hotspot.

`scripts/run-laptop.sh` reads its configuration **from the environment only**. The platform's
`laptop/start-mac.sh` sources `~/.config/isr-kit/station.env` and then calls it; see
`deploy/station.env.example` for the keys. The script:
- creates `.venv-laptop` on first run and installs `requirements-desktop.txt`, which includes PyTorch;
- binds `0.0.0.0:8130` and runs in the foreground;
- exits 0 on SIGTERM or Ctrl+C, after publishing a retained `stopped` heartbeat;
- keeps the Mac awake while it runs.

Defaults when a variable is unset: `TAK_HOST=isr-pi.local`, `TAK_PORT=8087`,
`MQTT_HOST=isr-pi.local`, `MQTT_PORT=1883`, `MQTT_USER=isrkit`, `MQTT_PASS` (it must be set),
`VLM_URL=http://localhost:11434`, `VLM_MODEL=qwen3-vl:8b`, `CHANGE_DETECT_PROFILE=laptop`.

```bash
git clone <this repo> && cd change-detect
MQTT_PASS=<isrkit password> scripts/run-laptop.sh      # or via the platform's laptop/start-mac.sh
open http://isr-hq.local:8130/
```

- **Laptop profile.** Stills stay at **full resolution**: GoPro 27 MP, 360 at 5.7K+, 4K drone
  frames. Alignment runs on a 2000 px copy and is rescaled, so it stays fast.
- **YOLO on the laptop** uses PyTorch with the accelerator chosen automatically (CUDA, then Apple
  MPS, then CPU). It runs **yolo11m** on a GPU/MPS and yolo11n on CPU. The weights download once
  into `~/change-detect-data/models/`, so do that while you have internet.
- **Offline AI suggestions:** install [Ollama](https://ollama.com) on the HQ Mac and run
  `ollama pull qwen3-vl:8b` (about 6 GB, fine on 64 GB). Leave `VLM_BACKEND=auto`: suggestions then
  stay on the Mac, and the UI shows "local VLM (qwen3-vl:8b)". If Ollama isn't running and
  `GEMINI_API_KEY` is set, it falls back to Gemini cloud.
- **Pi address.** If mDNS fails on the hotspot, set `TAK_HOST` and `MQTT_HOST` to 192.168.4.1.
- **MQTT from the laptop** must use the `isrkit` user, because RabbitMQ's `guest` account only
  works on the Pi itself.
- **macOS prompts** "allow incoming connections" for Python the first time: allow it, so phones
  on the hotspot can reach the UI. The script keeps the Mac awake (`caffeinate`) while it runs.
- **Windows** (secondary): `powershell -ExecutionPolicy Bypass -File scripts\run-laptop.ps1`.
  It installs CUDA PyTorch if `nvidia-smi` is present.
- Reports go to `~/change-detect-data/reports/`.
- Don't run change-detect on the Pi and the Mac at the same time: both publish the same
  heartbeat topic.

## Run on a Mac (no hardware)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m changedetect --simulate --data-dir ./data     # http://localhost:8130
pytest -q
```

`--simulate` writes the bundled synthetic samples and preloads an analysis. One sample is an
overhead pass in which a car moved, a camouflaged hide appeared and some ground was dug. The other
is a 360 pair with a 37° heading change. TAK and MQTT connections fail quietly when no server is
running.

## Demo script (about 20 minutes)

**Kit:** GoPro (or phone), optionally the FPV drone or the 360 camera, a car or a pile of kit,
and the Pi running `bft-change-detect` with ATAK on a phone.

1. **Brief (2 min):** recce teams revisit a location and ask "what's different?". Comparing photos
   by eye is slow and easy to get wrong. Show the synthetic sample in the UI first.
2. **First pass:** walk or fly a short route. Stop at 2–3 marked points (a fence post, a tree) and
   take a still at each, or record one steady clip. Note the spot, height and heading.
3. **Make changes:** move a car 10–20 m, put out a bergen, cam net or ammo box, and scrape or dig
   a patch of ground. Optionally have someone stand in shot in one pass only.
4. **Second pass:** repeat the route from the **same marked points**, same height, same heading.
5. **Analyse:** copy the files to a laptop or phone and upload BEFORE and AFTER in the web UI.
   Show:
   - the alignment grade and the reason it gives. Then try a pair taken from a different spot:
     the tool refuses to call changes instead of guessing;
   - the swipe slider and heatmap, then the numbered boxes and crops;
   - YOLO labels ("vehicle appeared"), if the model is installed;
   - "Get AI suggestions" if online (point out the AI badge and that the crops go to Google).
6. **Report and share:** download the PDF, push the marker to ATAK (EXIF GPS or typed grid
   converted to lat/lon), and send the mesh summary to the Meshtastic nodes.
7. **Discuss:** what it missed, what it falsely flagged (shadows, wind-blown vegetation, parallax on
   tall objects), and why repeatable imagery matters.

### Tips for repeatable imagery

- **Same position, height and heading.** Mark the spot (a stake, a post, a GPS waypoint), note the
  compass heading, and for the drone hold the same altitude AGL and the same gimbal angle.
  Overhead (nadir) shots align best, because the ground is flat to the camera.
- **Same camera settings:** the same lens mode (GoPro *Linear* rather than *Wide/SuperView*, which
  distorts), the same zoom, and stills rather than video where possible.
- **Similar light:** about the same time of day, and avoid moving cloud shadows. Changing shadows
  are the biggest source of false alarms.
- **Overlap:** frame each shot so that at least 60 % of the scene appears in both passes, with
  plenty of fixed texture (tracks, fence lines, buildings).
- **Hold still:** blur kills feature matching. For video, pause for 2 s at each point.
- **360 camera:** export an equirectangular JPEG (2:1) from the camera app, and keep the camera at
  the same height on the same spot. The heading difference is corrected automatically.
- **Phone photos:** HEIC and JPEG both work. Leave location on for EXIF GPS, and use the 1× lens
  both times.
- **DJI O3/O4 recordings:** fly the same line twice at the same height. Hover or slow down over each
  point of interest, and give the approximate time in each clip. Turn off EIS/RockSteady
  horizon-lock differences between runs. These clips usually have no GPS, so type the position
  into the ATAK box.

## Limitations

- A homography assumes a flat scene or a pure camera rotation. From a different position, tall
  objects (trees, buildings, vehicles) show parallax that can be flagged as change. Repeat the
  viewpoint closely.
- Global illumination is normalised, but moving shadows, wet versus dry ground, wind in vegetation
  and seasonal change still produce false alarms.
- The marker position is the **camera's** EXIF GPS (or manual entry), not a geolocated position of
  each change. Imagery is not orthorectified.
- GoPro MP4 telemetry (GPMF GPS) and DJI SRT subtitles are not parsed, so enter the position
  manually for video. Insta360 `.insv` dual-fisheye files are not supported: export an
  equirectangular JPEG or MP4 from the Insta360 app first.
- Folder auto-split needs capture times (EXIF, or the file dates the browser sends) with one clear
  break between passes. Otherwise use the two boxes.
- Change "kind" (new / removed / surface change) is an edge- and colour-based heuristic. YOLO labels
  depend on the optional model and are weak on top-down imagery.
- On the Pi, processing works at about 1600 px (360 views at about 1024×768), so very small objects
  need closer shots, or the laptop profile.
- The ATAK marker is a single point per analysis, and one analysis runs at a time.
