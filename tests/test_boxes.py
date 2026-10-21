import numpy as np

from mot_track.boxes import clip_xyxy, iou_matrix, xywh_to_xyxy, xyxy_to_xywh


def test_conversion_roundtrip():
    xywh = np.array([[10, 20, 30, 40.0]])
    assert np.allclose(xyxy_to_xywh(xywh_to_xyxy(xywh)), xywh)
    assert np.allclose(xywh_to_xyxy(xywh), [[10, 20, 40, 60]])


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10.0]])
    b = np.array([[0, 0, 10, 10.0], [5, 0, 15, 10.0], [20, 20, 30, 30.0]])
    assert np.allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_iou_empty_and_degenerate():
    assert iou_matrix(np.zeros((0, 4)), np.ones((3, 4))).shape == (0, 3)
    zero = np.array([[5, 5, 5, 5.0]])
    assert iou_matrix(zero, zero)[0, 0] == 0.0


def test_clip():
    out = clip_xyxy(np.array([[-5, -5, 300, 120.0]]), 200, 100)
    assert np.allclose(out, [[0, 0, 200, 100]])
