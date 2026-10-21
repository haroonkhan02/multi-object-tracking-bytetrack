"""Drawing boxes and identities on frames."""

from __future__ import annotations

import cv2
import numpy as np

from .boxes import xywh_to_xyxy


def id_color(track_id: int) -> tuple[int, int, int]:
    """A stable BGR colour per identity."""
    rng = np.random.default_rng(int(track_id) * 2654435761 % (2**32))
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def draw_tracks(image: np.ndarray, rows: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Draw MOT rows (frame, id, x, y, w, h, ...) on a copy of `image`."""
    out = image.copy()
    for row in rows:
        x1, y1, x2, y2 = xywh_to_xyxy(row[2:6])[0].astype(int)
        color = id_color(int(row[1]))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = str(int(row[1]))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
        )
    return out
