"""Bounding-box helpers. Boxes are float arrays; `xywh` is top-left + size, `xyxy` is corners."""

from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = boxes.copy()
    out[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = boxes.copy()
    out[:, 2:] = boxes[:, 2:] - boxes[:, :2]
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of `xyxy` boxes, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def clip_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out
