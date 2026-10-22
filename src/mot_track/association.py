"""Matching tracks to detections: IoU cost + Hungarian assignment."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boxes import iou_matrix, xywh_to_xyxy


def iou_cost(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """1 - IoU between `xywh` track and detection boxes."""
    return 1.0 - iou_matrix(xywh_to_xyxy(track_boxes), xywh_to_xyxy(det_boxes))


def fuse_score(cost: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """ByteTrack's score fusion: similarity = IoU * detection score.

    A high-IoU match to a 0.3-confidence box then costs more than one to a 0.9 box, which helps
    when two detections overlap the same predicted track.
    """
    if cost.size == 0:
        return cost
    return 1.0 - (1.0 - cost) * np.asarray(scores)[None, :]


def linear_assignment(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost matching; pairs costing more than `max_cost` are rejected.

    Returns (matches, unmatched rows, unmatched columns).
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= max_cost]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    unmatched_r = [r for r in range(n_rows) if r not in matched_r]
    unmatched_c = [c for c in range(n_cols) if c not in matched_c]
    return matches, unmatched_r, unmatched_c
