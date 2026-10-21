"""Tuning / validation split.

MOT17's test annotations are private, so I evaluate on the public training sequences. To keep
tracker hyper-parameters from being tuned on the frames they are scored on, every sequence is
cut in half: frames in the first half are for tuning, frames in the second half for the final
numbers (the protocol used by the ByteTrack paper for its ablations).
"""

from __future__ import annotations

import numpy as np


def half_split(length: int) -> dict[str, tuple[int, int]]:
    """Inclusive 1-based frame ranges for the tuning (first) and validation (second) half."""
    mid = length // 2
    return {"tune": (1, mid), "val": (mid + 1, length)}


def frame_range(split: str, length: int) -> tuple[int, int]:
    if split == "full":
        return 1, length
    ranges = half_split(length)
    if split not in ranges:
        raise ValueError(f"unknown split {split!r}; use 'tune', 'val' or 'full'")
    return ranges[split]


def select_frames(rows: np.ndarray, first: int, last: int) -> np.ndarray:
    """Keep rows whose frame number (column 0) is within [first, last]."""
    if len(rows) == 0:
        return rows
    keep = (rows[:, 0] >= first) & (rows[:, 0] <= last)
    return rows[keep]
