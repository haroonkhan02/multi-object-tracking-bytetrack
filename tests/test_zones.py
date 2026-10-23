import numpy as np

from mot_track.zones import feet_in_zone, zone_alerts

ZONE = (0.0, 50.0, 100.0, 100.0)


def walk(tid, frames, y_feet=80.0):
    return [[f, tid, 40.0, y_feet - 20.0, 10.0, 20.0, 1.0] for f in frames]


def test_feet_in_zone_uses_bottom_centre():
    rows = np.array(walk(1, [1]) + walk(2, [1], y_feet=40.0))
    assert list(feet_in_zone(rows, ZONE)) == [True, False]


def test_one_alert_per_visit_after_min_dwell():
    rows = np.array(walk(1, range(1, 31)))  # 30 frames at 10 FPS = 3 s in the zone
    assert zone_alerts(rows, ZONE, fps=10, min_dwell_s=1.0) == [(1, 10)]


def test_short_visit_and_short_gap():
    short = walk(1, range(1, 6))  # 0.5 s: no alert
    gappy = walk(2, list(range(1, 6)) + list(range(8, 14)))  # 2-frame gap < 0.5 s grace
    alerts = zone_alerts(np.array(short + gappy), ZONE, fps=10, min_dwell_s=1.0, grace_s=0.5)
    assert alerts == [(2, 10)]


def test_id_switch_inside_zone_duplicates_the_alert():
    rows = np.array(walk(1, range(1, 16)) + walk(2, range(16, 31)))
    assert len(zone_alerts(rows, ZONE, fps=10, min_dwell_s=1.0)) == 2


def test_score_alerts_counts_duplicates_and_misses():
    from mot_track.zones import score_alerts

    gt = np.array([r + [1, 1.0] for r in walk(1, range(1, 31))])
    gt = np.vstack([gt, [r + [1, 1.0] for r in walk(2, range(1, 31), y_feet=40.0)]])
    # Person 1 is tracked as two IDs (switch at frame 16): one correct alert, one duplicate.
    trk = np.array(walk(7, range(1, 16)) + walk(8, range(16, 31)))
    s = score_alerts(gt, trk, np.zeros((0, 9)), ZONE, fps=10)
    assert s == {"visits": 1, "alerts": 2, "correct": 1, "duplicate": 1, "false": 0, "missed": 0}
