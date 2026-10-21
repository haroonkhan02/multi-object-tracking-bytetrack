"""Reading and writing MOTChallenge text files.

Every line is `frame, id, x, y, w, h, conf, class, visibility` (ground truth) or
`frame, id, x, y, w, h, conf, -1, -1, -1` (tracker output); boxes are top-left + size in pixels.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CLASS_DISTRACTORS, CLASS_PEDESTRIAN

COLUMNS = ("frame", "id", "x", "y", "w", "h", "conf", "cls", "vis")


@dataclass(frozen=True)
class SeqInfo:
    name: str
    img_dir: str
    fps: float
    length: int
    width: int
    height: int
    img_ext: str


def read_seqinfo(path: Path) -> SeqInfo:
    parser = configparser.ConfigParser()
    parser.read(path)
    s = parser["Sequence"]
    return SeqInfo(
        name=s["name"],
        img_dir=s.get("imDir", "img1"),
        fps=float(s["frameRate"]),
        length=int(s["seqLength"]),
        width=int(s["imWidth"]),
        height=int(s["imHeight"]),
        img_ext=s.get("imExt", ".jpg"),
    )


def load_mot_txt(path: Path) -> np.ndarray:
    """Load a MOT text file as a float array with 9 columns (missing ones filled with -1)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, len(COLUMNS)))
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    out = np.full((len(data), len(COLUMNS)), -1.0)
    n = min(data.shape[1], len(COLUMNS))
    out[:, :n] = data[:, :n]
    return out


def split_gt(gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split MOT17 ground truth into (evaluated pedestrians, distractor regions).

    Rows with conf == 0 are ignored by the benchmark. Tracker boxes that match a distractor
    (person on vehicle, static person, reflection) are discarded instead of counted as false
    positives, so the distractors are returned separately.
    """
    valid = gt[:, 6] > 0
    pedestrians = gt[valid & (gt[:, 7] == CLASS_PEDESTRIAN)]
    distractors = gt[valid & np.isin(gt[:, 7], CLASS_DISTRACTORS)]
    return pedestrians, distractors


def write_results(path: Path, rows: np.ndarray) -> None:
    """Write tracker output rows (frame, id, x, y, w, h, conf) in MOTChallenge format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=float).reshape(-1, 7)
    with path.open("w") as f:
        for r in rows[np.lexsort((rows[:, 1], rows[:, 0]))]:
            f.write(
                f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},"
                f"{r[6]:.4f},-1,-1,-1\n"
            )


def by_frame(rows: np.ndarray) -> dict[int, np.ndarray]:
    """Group rows by frame number."""
    out: dict[int, np.ndarray] = {}
    if len(rows) == 0:
        return out
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    frames, starts = np.unique(rows[:, 0].astype(int), return_index=True)
    for frame, chunk in zip(frames, np.split(rows, starts[1:]), strict=True):
        out[int(frame)] = chunk
    return out
