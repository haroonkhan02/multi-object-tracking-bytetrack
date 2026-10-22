import numpy as np
import pytest

from mot_track.dataset import Sequence
from mot_track.evaluate import prepare_split, run_trackeval
from mot_track.mot_format import load_mot_txt, write_results
from mot_track.track import track_sequence, tracker_fps

pytest.importorskip("trackeval")


@pytest.fixture
def tracked(mot_root, tmp_path):
    seq = Sequence("MOT17-02", root=mot_root)
    pub = seq.public_detections()
    dets = pub[:, [0, 2, 3, 4, 5, 6]]
    rows, seconds = track_sequence(seq, dets, "bytetrack")
    out = tmp_path / "results" / "byte"
    write_results(out / "MOT17-02.txt", rows)
    return mot_root, out, rows, seconds


def test_track_sequence_one_identity_per_frame(tracked):
    _, _, rows, seconds = tracked
    assert len(rows) == 10
    assert set(rows[:, 1]) == {1}
    assert tracker_fps({"s": {"frames": 10, "seconds": seconds}}) > 0


def test_prepare_split_renumbers_frames(tracked, tmp_path):
    root, out, _, _ = tracked
    work = tmp_path / "work"
    prepare_split(work, ["MOT17-02"], "val", {"byte": out}, root=root)
    gt = load_mot_txt(work / "gt" / "MOT17-02" / "gt" / "gt.txt")
    res = load_mot_txt(work / "trackers" / "byte" / "data" / "MOT17-02.txt")
    assert gt[:, 0].min() == 1 and gt[:, 0].max() == 5
    assert res[:, 0].min() == 1 and res[:, 0].max() == 5
    assert "seqLength=5" in (work / "gt" / "MOT17-02" / "seqinfo.ini").read_text()


def test_trackeval_scores_half_detected_scene(tracked, tmp_path):
    root, out, _, _ = tracked
    work = tmp_path / "work"
    prepare_split(work, ["MOT17-02"], "full", {"byte": out}, root=root)
    m = run_trackeval(work, ["byte"], ["MOT17-02"])["byte"]["COMBINED_SEQ"]
    # Pedestrian 1 is tracked perfectly, pedestrian 2 is never detected.
    assert m["FP"] == 0 and m["IDSW"] == 0
    assert m["FN"] == 10
    assert m["MOTA"] == pytest.approx(50.0)
    assert m["IDF1"] == pytest.approx(100 * 2 * 10 / 30)
    assert m["AssA"] > 90  # Kalman-smoothed boxes are not pixel-exact at high alpha
    assert np.isfinite(m["HOTA"])
