# change-detect

Compare **before** and **after** imagery of the same area and highlight what changed: a vehicle that moved, a new
structure, disturbed ground. It runs entirely on a
Raspberry Pi 5 CPU (OpenCV, no AI accelerator) with a web UI on **port 8130**.

> Training and demonstration tool. Every change it reports is a machine guess: confirm each one by eye.
> It is defensive and observational only.

## What it does

1. **Inputs** (web UI): upload two stills or two videos, or pick a bundled sample. Stills can be
   JPEG, PNG or iPhone HEIC. Videos can be MP4/MOV, including GoPro Hero 4 clips (no GPS) and DJI
   O3/O4 onboard recordings (H.264/H.265, 4K, usually no GPS). For videos it samples frames from both clips and picks the
   sharpest pair that overlaps best; you can give an approximate time in each clip. EXIF GPS,
   capture time and camera are read where present.
   **Folders:** drop a folder on Before and another on After, or drop a whole SD card / DCIM
   folder in Details and it is split into two passes at the largest gap in capture time. Matching shots are
   paired by GPS (within 40 m), by position in the sequence and by feature matching, and each pair
   is analysed. Sidecar files (`.LRV`, `.THM`, `._*`) are skipped.
2. **360** inputs come in two kinds:
   - **Equirectangular** (2:1, or tagged `GPano:ProjectionType=equirectangular`).
   - **Raw dual-fisheye from the original Insta360 ONE**: `.insp` stills are read as JPEG and
     `.insv` video as MP4. The side-by-side two-circle layout is checked first, and a file that
     fails the check is rejected with a message suggesting the iPhone or GoPro. Each lens is
     unwarped to a panorama with an equidistant model (`FISHEYE_FOV_DEG`, default 200).

   For both kinds it estimates the heading difference between the two panoramas, then compares
   four matching perspective views (front/right/back/left).
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
   OpenAI-compatible endpoint first (Ollama on the HQ Mac, `qwen3-vl:8b-instruct`, offline), then Gemini
   if `GEMINI_API_KEY` is set, otherwise off. `openai_compat`, `gemini` or `off` forces one backend.
8. **Optional ATAK**: pushes a CoT "change detected" marker (`a-u-G`, unknown ground) with the
   summary. The position is a manually entered lat/lon, else EXIF GPS, else the OP position
   `SENSOR_LAT`/`SENSOR_LON`. With none of these the marker is refused: it is never sent at 0,0.
   **Mesh**: sends a short summary text over the Meshtastic mesh via a separate mesh bridge service (MQTT `meshbus/tx`).
   - Results from the bundled synthetic samples are labelled: CoT remarks start `SIMULATED: ` and
     mesh texts start `[SIM] `. Your own imagery carries no prefix.
   - Rate limits: at most one mesh text every 30 s, and one CoT event per marker uid every 5 s.
     The UI says when to try again.
   - **TAK connection.** OpenTAKServer drops CoT from senders that are not a registered device.
     So change-detect keeps one TCP connection open from startup and reconnects with backoff.
     After each connect it waits 3 s, then registers. Markers go out on that connection once
     registration is done; until then they queue.
     - It registers with a position report: uid `isrkit-change-detect-<ISR_HOST>`, callsign
       `CHANGE <ISR_HOST>`, type `a-f-G-E-S`, group Cyan. The report is re-sent every 60 s.
     - The report uses `SENSOR_LAT`/`SENSOR_LON`. If those are unset, it uses a "no position"
       point that ATAK never draws.
     - The uid and callsign are stable, so a restart takes back the same identity. OpenTAKServer
       silently refuses a callsign another uid already holds.
     - Change markers are built so OpenTAKServer also stores them and shows them on its web map.
       They have type `a-u-G`, no `<takv>`, no contact endpoint, a real position (never 0,0), and
       a single `<link>` to the registration uid.

The page uses the shared ISTAR kit look (`static/isr-ui.css`, `isr-ui.js` and `fonts/`, vendored
unchanged from `ui-kit/`). The header has one status pill (Ready / Degraded / Offline); click it
for the server mode, AI source, YOLO, TAK and mesh, which are optional for the demo. Settings,
model names, file paths and diagnostics are under Details; past reports under Report history.

Every 10 s it publishes a retained heartbeat to `isrkit/heartbeat/change-detect`, so the Pi
launcher can link to it wherever it runs. The payload is
`{"tool":"change-detect","host":"isr-hq","url":"http://isr-hq.local:8130/","mode":"mps","status":"ok","ts":…}`.
- `host` is `ISR_HOST` if set, else the lowercased short hostname.
- `mode` is `cpu`, `mps`, `cuda` or `simulate`.
- `status` is `ok`, or `degraded` when ATAK markers are stuck in the queue or a mesh send failed.
- On a clean exit (SIGTERM) it publishes a retained `stopped`. If the process dies uncleanly, the
  MQTT last-will publishes `error`.

## Install on the Pi

```bash
git clone <this repo> && cd change-detect
sudo bash deploy/install.sh
sudo pi-profiles change-detect          # this repo ships only its own profiles.d/change-detect.conf
```

Combined profiles (for example with OpenTAKServer and the mesh bridge, so markers reach ATAK)
belong to the pi-platform repo, not this one.

- Code: `/opt/change-detect` (venv in `.venv`). Data and reports: `/var/lib/change-detect`.
  Config: `/etc/change-detect.env` (created from `deploy/change-detect.env.example`, root-only).
- The service runs as the `changedetect` system user and is **not** enabled at boot: pi-profiles
  starts it.
- Logs: `journalctl -u change-detect -f`.

### Configuration (`/etc/change-detect.env`)

| Variable | Default | Purpose |
|---|---|---|
| `TAK_HOST` / `TAK_PORT` | `127.0.0.1` / `8087` | CoT TCP output: one long-lived, registered connection (see below). If the TAK server is down, markers queue and it reconnects with backoff. |
| `MQTT_HOST` / `MQTT_PORT` / `MQTT_USER` / `MQTT_PASS` | `127.0.0.1` / `1883` / `guest` / `guest` | Publishes to `meshbus/tx`. It never opens the radio itself. |
| `VLM_BACKEND` | `auto` | `auto` (local VLM → Gemini → off), `openai_compat`, `gemini` or `off`. |
| `VLM_URL` / `VLM_MODEL` | `http://localhost:11434` / `qwen3-vl:8b-instruct` | **Bare** Ollama base URL, with no path (the tool appends `/v1/chat/completions`). On the Pi, set `http://isr-hq.local:11434`. |
| `GEMINI_API_KEY` | empty | Empty means no cloud. Set it to allow Gemini suggestions. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any current Gemini model that accepts images. |
| `YOLO_BACKEND` | `auto` | `auto` uses ultralytics (PyTorch) if installed (laptop), else ONNX via OpenCV (Pi). `off` disables YOLO. |
| `YOLO_MODEL` | `/var/lib/change-detect/models/yolo11n.onnx` | ONNX model for the Pi path. |
| `YOLO_WEIGHTS` | `yolo11m.pt` on MPS/CUDA, `yolo11n.pt` on CPU | Ultralytics weights for the laptop path. |
| `CHANGE_DETECT_PROFILE` | `pi` | `pi` downscales stills to 1600 px. `laptop` keeps full resolution. |
| `HEARTBEAT` / `PUBLIC_URL` | `1` / auto | Heartbeat on/off, and the URL it advertises. |
| `ISR_HOST` | short hostname | The heartbeat `host` name (and the `<host>.local` URL). |
| `SENSOR_LAT` / `SENSOR_LON` | unset | OP position, used for ATAK/mesh when images have no GPS. `0,0` counts as unset. |
| `YOLO_ONNX_URL` | unset | Where `deploy/install.sh` downloads `yolo11n.onnx` from, if it isn't in the repo's `models/`. |
| `CHANGE_DETECT_ARGS` | empty | e.g. `--simulate` or `--no-cot`. |

### Optional: YOLO nano labels

YOLO is not bundled: the weights come under Ultralytics' AGPL-3.0 licence, and the Pi needs no
PyTorch. On a laptop with internet access:

```bash
# on the Mac, while online
scripts/run-laptop.sh --setup-only --export-onnx    # writes ~/change-detect-data/models/yolo11n.onnx
# copy it into the Pi's checkout, then (re-)run the installer, which copies it into place
scp ~/change-detect-data/models/yolo11n.onnx pi@isr-pi.local:change-detect/models/
ssh pi@isr-pi.local 'sudo change-detect/deploy/install.sh && sudo systemctl restart change-detect'
```

`deploy/install.sh` runs `python -m changedetect.prefetch`. That copies a `yolo11n.onnx` from the
repo's `models/` dir (or downloads `YOLO_ONNX_URL`) and reports any model that is missing. If the
model is missing, the UI shows "YOLO: model missing" and analysis carries on without labels.

COCO classes are trained on ground-level photos. Expect good results on GoPro and phone shots,
and poor results on top-down drone imagery from height.

### Optional: Gemini cloud mode

> **Cloud mode sends image crops to Google.** Use demo footage only. Never use real or sensitive
> imagery. With no key, or with no internet, everything else works offline.

## Running on a laptop

The Pi `isr-pi.local` (hotspot 192.168.4.1) stays the hub: hotspot, OpenTAKServer, mesh bridge,
MQTT broker and launcher. change-detect runs on the **HQ Mac** `isr-hq.local` (which also
runs Ollama) as the analysis station. The Mac joins the Pi's hotspot.

`scripts/run-laptop.sh` reads its configuration **from the environment only**. The platform's
`laptop/start-mac.sh` sources `~/.config/isr-kit/station.env` and then calls it; see
`deploy/station.env.example` for the keys. The script:
- creates `.venv-laptop` on first run and installs `requirements-desktop.txt`, which includes PyTorch;
- with `--setup-only`, downloads every model needed offline and exits: the YOLO weights
  (yolo11m and yolo11n), and `ollama pull $VLM_MODEL` when Ollama runs on this Mac.
  **Run it once while online, before joining the hotspot.**
- binds `0.0.0.0:8130` and runs in the foreground;
- exits 0 on SIGTERM or Ctrl+C, after publishing a retained `stopped` heartbeat;
- keeps the Mac awake while it runs.

Defaults when a variable is unset: `TAK_HOST=isr-pi.local`, `TAK_PORT=8087`,
`MQTT_HOST=isr-pi.local`, `MQTT_PORT=1883`, `MQTT_USER=isrkit`, `MQTT_PASS` (it must be set),
`VLM_URL=http://localhost:11434`, `VLM_MODEL=qwen3-vl:8b-instruct`, `CHANGE_DETECT_PROFILE=laptop`.

```bash
git clone <this repo> && cd change-detect
scripts/run-laptop.sh --setup-only                     # once, while online
MQTT_PASS=<isrkit password> scripts/run-laptop.sh      # or via the platform's laptop/start-mac.sh
open http://isr-hq.local:8130/
```

- **Laptop profile.** Stills stay at **full resolution**: GoPro 27 MP, 360 at 5.7K+, 4K drone
  frames. Alignment runs on a 2000 px copy and is rescaled, so it stays fast.
- **YOLO on the laptop** uses PyTorch with the accelerator chosen automatically (CUDA, then Apple
  MPS, then CPU). It runs **yolo11m** on a GPU/MPS and yolo11n on CPU. `--setup-only` puts the
  weights in `~/change-detect-data/models/`.
  - If yolo11m is missing, the app falls back to yolo11n when that is present.
  - Offline with no weights at all, the UI says "YOLO: model missing" and labels are off; it does not hang.
  - Online with weights missing, it downloads them in the background.
- **Offline AI suggestions:** install [Ollama](https://ollama.com) on the HQ Mac and run
  `ollama pull qwen3-vl:8b-instruct` (about 6 GB).
  Use the **instruct** build. Plain `qwen3-vl:8b` is a thinking build: it reasons even when told
  not to and sometimes returns empty text. An empty answer counts as a failure and falls back to
  the next backend. Replies are capped at 96 tokens and requested as plain text: no JSON
  `format`, which MLX-runner models reject with HTTP 501. Leave `VLM_BACKEND=auto`: suggestions then
  stay on the Mac, and the UI shows "local VLM (qwen3-vl:8b-instruct)". If Ollama isn't running and
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
overhead pass in which a car moved, a covered shelter appeared and some ground was dug. The other
is a 360 pair with a 37° heading change. TAK and MQTT connections fail quietly when no server is
running.

## Demo script (about 20 minutes)

**Kit:** iPhone or GoPro Hero 4, optionally the FPV drone (DJI O3/O4 recordings) or the
Insta360 ONE, a car or a few large objects, and the Pi running a pi-platform profile that includes
change-detect and OpenTAKServer, with ATAK on a phone.

1. **Brief (2 min):** survey and inspection teams revisit a location and ask "what's different?". Comparing photos
   by eye is slow and easy to get wrong. Show the synthetic sample in the UI first.
2. **First pass:** walk or fly a short route. Stop at 2–3 marked points (a fence post, a tree) and
   take a still at each, or record one steady clip. Note the spot, height and heading.
3. **Make changes:** move a car 10–20 m, put out a rucksack, a tarp or a box, and scrape or dig
   a patch of ground. Optionally have someone stand in shot in one pass only.
4. **Second pass:** repeat the route from the **same marked points**, same height, same heading.
5. **Analyse:** copy the files to a laptop or phone, drop them on Before and After and press Compare.
   Show:
   - the alignment grade and the reason it gives. Then try a pair taken from a different spot:
     the tool refuses to call changes instead of guessing;
   - the swipe slider and heatmap, then the numbered boxes and crops;
   - YOLO labels ("vehicle appeared"), if the model is installed;
   - "Describe with AI" (the status pill and Details show whether it is local or Gemini, whose crops go to Google).
6. **Report and share:** open the report PDF, send the marker to ATAK (EXIF GPS, or a
   position typed under Result details), and send the mesh summary to the Meshtastic nodes.
7. **Discuss:** what it missed, what it falsely flagged (shadows, wind-blown vegetation, parallax on
   tall objects), and why repeatable imagery matters.

### Tips for repeatable imagery

- **Same position, height and heading.** Mark the spot (a stake, a post, a GPS waypoint), note the
  compass heading, and for the drone hold the same altitude AGL and the same gimbal angle.
  Overhead (nadir) shots align best, because the ground is flat to the camera.
- **Same camera settings:** the same lens mode. On the GoPro Hero 4, prefer *Medium* over *Wide*
  field of view, because the wide fisheye bends straight lines and hurts alignment. Use the same
  mode on both passes, the same zoom, and stills rather than video where possible.
- **Similar light:** about the same time of day, and avoid moving cloud shadows. Changing shadows
  are the biggest source of false alarms.
- **Overlap:** frame each shot so that at least 60 % of the scene appears in both passes, with
  plenty of fixed texture (tracks, fence lines, buildings).
- **Hold still:** blur kills feature matching. For video, pause for 2 s at each point.
- **360 camera (Insta360 ONE):** raw `.insp` and `.insv` files can go straight in; an
  equirectangular export also works. Keep the camera at the same height on the same spot, upright,
  on a stick or tripod. The heading difference is corrected automatically. The person holding the
  camera appears in the nadir and differs between passes, so expect some noise low in the
  side views.
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
- GoPro Hero 4 has no GPS, and DJI SRT subtitles are not parsed, so enter the position manually
  for those.
- **The Insta360 ONE unwarp is uncalibrated.** It uses an equidistant model with a fixed field of
  view, does no seam blending, and its orientation has not been checked against real ONE files.
  Before and after go through the same mapping, so the comparison stays consistent, but the
  views can look slightly bent or rotated, and changes on the lens seam (the 90° and 270° views)
  are less reliable. It has only been tested on synthetic dual-fisheye frames. If real files look
  wrong, tune `FISHEYE_FOV_DEG` or use an equirectangular export.
- Folder auto-split needs capture times (EXIF, or the file dates the browser sends) with one clear
  break between passes. Otherwise use the two boxes.
- Change "kind" (new / removed / surface change) is an edge- and colour-based heuristic. YOLO labels
  depend on the optional model and are weak on top-down imagery.
- On the Pi, processing works at about 1600 px (360 views at about 1024×768), so very small objects
  need closer shots, or the laptop profile.
- The ATAK marker is a single point per analysis, and one analysis runs at a time.

## License

Copyright (C) 2026 Jacob Haynes. Licensed under the GNU AGPL v3 (see [LICENSE](LICENSE)). Uses Ultralytics YOLO (AGPL-3.0); model weights are subject to Ultralytics' licence.
