import cv2
import numpy as np
import pytest

from changedetect import media, pipeline, synthetic


def test_exif_gps_and_time(samples):
    d, _ = samples
    m = media.read_exif(d / "synthetic_before.jpg")
    assert m.lat == pytest.approx(synthetic.DEMO_LAT, abs=1e-5)
    assert m.lon == pytest.approx(synthetic.DEMO_LON, abs=1e-5)
    assert m.time == "2026-11-14T09:12:30"
    assert not m.equirect_tag


def test_no_exif_is_fine(tmp_path):
    p = tmp_path / "plain.png"
    cv2.imwrite(str(p), np.zeros((50, 80, 3), np.uint8))
    m = media.read_exif(p)
    assert m.lat is None and m.time is None and m.width == 80


def test_unsupported_type():
    with pytest.raises(media.MediaError):
        media.kind_of("clip.insv")


def test_equirect_tag_detected(samples):
    d, _ = samples
    assert media.read_exif(d / "synthetic360_before.jpg").equirect_tag


def test_yaw_offset_estimate():
    b, a, truth = synthetic.make_equirect_pair(w=2048)
    yaw, conf = media.estimate_yaw_offset(b, a)
    assert yaw == pytest.approx(truth["yaw_offset_deg"], abs=1.0)
    assert conf > 0.3


def test_perspective_crop_shape():
    eq = np.zeros((512, 1024, 3), np.uint8)
    eq[:, 512:520] = 255  # vertical line at lon 0 (image centre)
    view = media.equirect_to_perspective(eq, 0, 0, 90, 320, 240)
    assert view.shape == (240, 320, 3)
    assert view[:, 150:170].mean() > view[:, :100].mean()


def test_360_pipeline_finds_both_changes(samples, tmp_path):
    d, index = samples
    s = index[1]
    r = pipeline.analyze(d / s["before"], d / s["after"], tmp_path)
    assert r["mode"] == "360"
    assert r["yaw_offset"]["deg"] == pytest.approx(37.0, abs=1.5)
    by_view = {v["name"]: len(v["changes"]) for v in r["views"]}
    assert by_view["360 right (90°)"] >= 1  # new hide
    assert by_view["360 left (270°)"] >= 1  # car gone
    assert by_view["360 front (0°)"] == 0 and by_view["360 back (180°)"] == 0


def _write_video(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("no MJPG video writer in this OpenCV build")
    for f in frames:
        vw.write(f)
    vw.release()


def test_video_best_matching_frames(flat_pair, tmp_path):
    before, after, _ = flat_pair
    small_b = cv2.resize(before, (720, 540))
    small_a = cv2.resize(after, (720, 540))
    rng = np.random.default_rng(1)
    junk = lambda: cv2.GaussianBlur((rng.random(small_b.shape) * 255).astype(np.uint8), (0, 0), 3)
    # before video: junk, then the scene; after video: scene in the middle
    _write_video(tmp_path / "b.avi", [junk() for _ in range(10)] + [small_b] * 10)
    _write_video(tmp_path / "a.avi", [junk() for _ in range(8)] + [small_a] * 6 + [junk() for _ in range(8)])
    r = pipeline.analyze(tmp_path / "b.avi", tmp_path / "a.avi", tmp_path / "out")
    assert r["video"]["match_inliers"] > 50
    assert 0.8 <= r["video"]["after_time_s"] <= 1.4
    assert r["inputs"]["before"]["kind"] == "video"
    assert r["alignment_ok"] and r["changes_total"] >= 3


@pytest.mark.parametrize("fourcc", ["hvc1", "avc1"])
def test_dji_style_mp4_decodes(tmp_path, fourcc):
    """DJI O3/O4 onboard recordings are H.264/H.265 MP4 with no GPS."""
    p = tmp_path / f"DJI_0001_{fourcc}.MP4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), 30, (1280, 720))
    if not vw.isOpened():
        pytest.skip(f"no {fourcc} encoder in this OpenCV build")
    f = np.zeros((720, 1280, 3), np.uint8)
    for i in range(20):
        f[:] = i * 10
        vw.write(f)
    vw.release()
    frames = media.sample_video(p, n=4, max_dim=None)
    assert len(frames) >= 3 and frames[0][1].shape == (720, 1280, 3)
    assert media.kind_of(p) == "video"
