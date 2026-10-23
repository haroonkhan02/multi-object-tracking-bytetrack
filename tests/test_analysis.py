import numpy as np
import pytest

from mot_track.analysis import (
    best_det_score,
    breakdown,
    clear_match,
    drop_distractor_matches,
    max_overlap,
    summarise,
)


def gt_row(frame, pid, x, vis=1.0, y=0.0, w=10.0, h=20.0):
    return [frame, pid, x, y, w, h, 1, 1, vis]


def trk_row(frame, tid, x, y=0.0, w=10.0, h=20.0):
    return [frame, tid, x, y, w, h, 1.0]


def test_perfect_tracking_has_no_switches():
    gt = np.array([gt_row(f, 1, 5 * f) for f in range(1, 6)])
    trk = np.array([trk_row(f, 7, 5 * f) for f in range(1, 6)])
    m = clear_match(gt, trk)
    assert (m.track_id == 7).all() and not m.switch.any()


def test_switch_after_occlusion_records_the_gap():
    gt = np.array([gt_row(f, 1, 0) for f in range(1, 9)])
    # Track 3 follows frames 1-2, nothing in frames 3-5, then a new track 4.
    trk = np.array([trk_row(f, 3, 0) for f in (1, 2)] + [trk_row(f, 4, 0) for f in (6, 7, 8)])
    m = clear_match(gt, trk)
    assert m.switch.sum() == 1
    assert m.gap[m.switch][0] == 3
    assert list(m.track_id) == [3, 3, -1, -1, -1, 4, 4, 4]


def test_continuing_match_wins_over_better_iou():
    # Frame 2: the person's own track overlaps less than a neighbour's track, but CLEAR keeps
    # the existing pair while its IoU stays >= 0.5, so there is no switch.
    gt = np.array([gt_row(1, 1, 0), gt_row(2, 1, 0)])
    trk = np.array([trk_row(1, 5, 0), trk_row(2, 5, 3), trk_row(2, 6, 0.5)])
    m = clear_match(gt, trk)
    assert list(m.track_id) == [5, 5] and not m.switch.any()


def test_tracker_box_on_distractor_is_dropped():
    gt = np.array([gt_row(1, 1, 0)])
    distractors = np.array([[1, 9, 100, 0, 10, 20, 1, 7, 1.0]])
    trk = np.array([trk_row(1, 1, 0), trk_row(1, 2, 100)])
    kept = drop_distractor_matches(trk, gt, distractors)
    assert list(kept[:, 1]) == [1]


def test_max_overlap_and_detector_scores():
    gt = np.array([gt_row(1, 1, 0), gt_row(1, 2, 5), gt_row(1, 3, 100), gt_row(2, 1, 0)])
    assert max_overlap(gt) == pytest.approx([1 / 3, 1 / 3, 0.0, 0.0])
    dets = np.array([[1, 0, 0, 10, 20, 0.4], [1, 1, 0, 10, 20, 0.9], [2, 50, 0, 10, 20, 0.8]])
    assert best_det_score(gt, dets) == pytest.approx([0.9, 0.0, 0.0, 0.0])


def test_breakdown_and_summary_on_synthetic_data():
    n = 4
    data = {
        "seq": np.array(["MOT17-02"] * n),
        "vis": np.array([0.1, 0.1, 0.9, 0.9]),
        "height": np.array([40.0, 40.0, 300.0, 300.0]),
        "overlap": np.zeros(n),
        "det": np.array([0.0, 0.15, 0.5, 0.9]),
        "trackers": {
            "T": {
                "matched": np.array([False, True, True, True]),
                "switch": np.array([False, False, True, False]),
                "gap": np.array([-1, -1, 0, -1]),
            }
        },
    }
    s = summarise(data)
    low, high = s["visibility"][0], s["visibility"][-1]
    assert low["boxes"] == 2 and low["recall T"] == 0.5 and low["recall dets >= 0.1"] == 0.5
    assert high["IDSW/1k T"] == 500.0 and "IDSW/1k any det" not in high
    assert s["switch_gaps"]["T"]["0"] == 1 and s["totals"]["T"]["IDSW"] == 1
    rows = breakdown(data["vis"], ((0.0, 1.01),), {"T": data["trackers"]["T"]["matched"]}, {})
    assert rows[0]["recall T"] == 0.75
