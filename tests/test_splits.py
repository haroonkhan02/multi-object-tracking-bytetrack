import numpy as np
import pytest

from mot_track.splits import frame_range, half_split, select_frames


def test_half_split_covers_all_frames_without_overlap():
    for length in (7, 600, 1050):
        s = half_split(length)
        assert s["tune"][0] == 1 and s["val"][1] == length
        assert s["tune"][1] + 1 == s["val"][0]


def test_frame_range_and_errors():
    assert frame_range("full", 10) == (1, 10)
    assert frame_range("val", 10) == (6, 10)
    with pytest.raises(ValueError):
        frame_range("test", 10)


def test_select_frames():
    rows = np.array([[f, 1, 0, 0, 1, 1, 1] for f in range(1, 11)], dtype=float)
    assert select_frames(rows, 6, 10)[:, 0].tolist() == [6, 7, 8, 9, 10]
    assert len(select_frames(np.zeros((0, 7)), 1, 5)) == 0
