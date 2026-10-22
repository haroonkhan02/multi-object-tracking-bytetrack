import numpy as np
import pytest

from mot_track.trackers import ByteConfig, ByteTrack, Sort, SortConfig, make_tracker


def walker(frame, x0=10.0, vx=5.0, y=20.0, score=0.9):
    return [x0 + vx * frame, y, 30.0, 80.0, score]


def run(tracker, frames):
    return [tracker.update(np.array(dets, float).reshape(-1, 5)) for dets in frames]


@pytest.mark.parametrize("name", ["sort", "bytetrack"])
def test_two_walkers_keep_their_ids(name):
    frames = [[walker(f), walker(f, x0=400, vx=-5)] for f in range(1, 40)]
    out = run(make_tracker(name), frames)
    ids_last = {int(r[0]) for r in out[-1]}
    assert len(ids_last) == 2
    all_ids = {int(r[0]) for rows in out for r in rows}
    assert all_ids == ids_last  # no identity switches or fragments


def test_sort_needs_min_hits_after_start():
    sort = Sort(SortConfig(min_hits=3))
    run(sort, [[walker(f)] for f in range(1, 6)])
    out = run(sort, [[walker(f), walker(f, x0=600, vx=0)] for f in range(6, 9)])
    assert [len(rows) for rows in out] == [1, 1, 2]


def test_sort_drops_low_scores_bytetrack_keeps_track_alive():
    # One walker; its score drops to 0.3 for five frames (partial occlusion).
    scores = [0.9] * 10 + [0.3] * 5 + [0.9] * 10
    frames = [[walker(f, score=s)] for f, s in enumerate(scores, start=1)]
    sort_out = run(Sort(SortConfig(max_age=1)), frames)
    byte_out = run(ByteTrack(), frames)
    # SORT never sees the 0.3 boxes, loses the track and starts a new identity.
    sort_ids = {int(r[0]) for rows in sort_out for r in rows}
    assert len(sort_ids) == 2
    assert all(len(rows) == 0 for rows in sort_out[10:15])
    # ByteTrack matches them in the second round: one identity, no gap.
    byte_ids = {int(r[0]) for rows in byte_out for r in rows}
    assert byte_ids == {1}
    assert all(len(rows) == 1 for rows in byte_out)


def test_bytetrack_recovers_identity_after_short_occlusion():
    frames = [[walker(f)] for f in range(1, 11)] + [[] for _ in range(8)]
    frames += [[walker(f)] for f in range(19, 30)]
    out = run(ByteTrack(ByteConfig(track_buffer=30)), frames)
    assert all(len(rows) == 0 for rows in out[10:18])
    assert {int(r[0]) for rows in out for r in rows} == {1}


def test_bytetrack_forgets_after_track_buffer():
    frames = [[walker(f)] for f in range(1, 11)] + [[] for _ in range(12)]
    frames += [[walker(f)] for f in range(23, 30)]
    out = run(ByteTrack(ByteConfig(track_buffer=10)), frames)
    assert {int(r[0]) for rows in out for r in rows} == {1, 2}


def test_bytetrack_track_buffer_scales_with_frame_rate():
    assert ByteTrack(ByteConfig(track_buffer=30), frame_rate=14).max_lost == 14


def test_output_filters_wide_boxes():
    out = run(ByteTrack(), [[[0, 0, 100, 40, 0.9]]])
    assert len(out[0]) == 0


def test_unknown_tracker():
    with pytest.raises(ValueError):
        make_tracker("deepsort")
