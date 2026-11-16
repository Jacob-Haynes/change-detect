# change-detect

Compare **before** and **after** imagery of the same area and highlight what changed. This is the
classic recce/ISR task: a vehicle that moved, a new hide, disturbed ground. It runs entirely on a
Raspberry Pi 5 CPU (OpenCV, no AI accelerator) with a web UI on **port 8130**.

> Training and demonstration tool. Every change it reports is a machine guess: confirm each one by eye.
> It is defensive and observational only.

## What it does

1. **Inputs** (web UI): upload two stills or two videos from a drone, GoPro or phone, or pick a
   bundled sample. For videos it samples frames from both clips and picks the sharpest pair that
   overlaps best (you can give an approximate time in each clip). EXIF GPS, capture time and camera
   are read where present.
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
7. **Optional AI suggestion** (Gemini): sends the two aligned crops of each change to Google and
   shows a one-line description labelled "AI suggestion".
8. **Optional ATAK**: pushes a CoT "change detected" marker (`a-u-G`, unknown ground) with the
   summary, using EXIF GPS or a manually entered lat/lon. **Mesh**: sends a short summary text over
   the Meshtastic mesh via the blufor-tracker bridge.

The header always shows which modes are active: `SIMULATE`, `AI: OFFLINE` or `AI: CLOUD (model)`,
YOLO on/off, TAK connection and mesh state.

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
| `GEMINI_API_KEY` | empty | Empty means offline. Set it to enable AI suggestions. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any current Gemini model that accepts images. |
| `YOLO_MODEL` | `/var/lib/change-detect/models/yolo11n.onnx` | Optional labelling model. |
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
- **Phone JPEGs:** use "Most Compatible" (JPEG) rather than HEIC, and leave location on for EXIF GPS.

## Limitations

- A homography assumes a flat scene or a pure camera rotation. From a different position, tall
  objects (trees, buildings, vehicles) show parallax that can be flagged as change. Repeat the
  viewpoint closely.
- Global illumination is normalised, but moving shadows, wet versus dry ground, wind in vegetation
  and seasonal change still produce false alarms.
- The marker position is the **camera's** EXIF GPS (or manual entry), not a geolocated position of
  each change. Imagery is not orthorectified.
- GoPro MP4 telemetry (GPMF GPS) and Insta360 `.insv` files are not parsed, and HEIC is not
  supported. Use JPEG stills or exported equirectangular JPEGs, and enter the position manually
  for video.
- Change "kind" (new / removed / surface change) is an edge- and colour-based heuristic. YOLO labels
  depend on the optional model and are weak on top-down imagery.
- Processing works at about 1600 px (360 views at 960×720), so very small objects need closer shots.
- The ATAK marker is a single point per analysis, and one analysis runs at a time.
