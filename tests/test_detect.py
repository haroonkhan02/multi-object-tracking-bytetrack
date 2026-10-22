from types import SimpleNamespace

import numpy as np

from mot_track.detect import boxes_from_result


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def test_boxes_from_result_converts_xyxy_to_xywh_rows():
    boxes = SimpleNamespace(xyxy=_Arr([[10, 20, 40, 100], [0, 0, 5, 5]]), conf=_Arr([0.9, 0.2]))
    rows = boxes_from_result(SimpleNamespace(boxes=boxes), frame=7)
    assert rows.shape == (2, 6)
    assert rows[0].tolist() == [7, 10, 20, 30, 80, np.float32(0.9)]
    assert rows[1, 5] == np.float32(0.2)


def test_boxes_from_result_empty():
    boxes = SimpleNamespace(xyxy=_Arr(np.zeros((0, 4))), conf=_Arr([]))
    assert boxes_from_result(SimpleNamespace(boxes=boxes), frame=1).shape == (0, 6)
