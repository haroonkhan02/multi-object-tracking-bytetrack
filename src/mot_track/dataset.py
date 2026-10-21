"""Access to the extracted MOT17 training sequences."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import IMAGE_DETECTOR, MOT17_DIR, TRAIN_SEQUENCES
from .mot_format import SeqInfo, load_mot_txt, read_seqinfo, split_gt
from .splits import frame_range, select_frames


@dataclass
class Sequence:
    """One MOT17 sequence: images, ground truth and the public detections."""

    name: str
    root: Path = MOT17_DIR
    detector: str = IMAGE_DETECTOR
    info: SeqInfo = field(init=False)

    def __post_init__(self) -> None:
        self.info = read_seqinfo(self.dir / "seqinfo.ini")

    @property
    def dir(self) -> Path:
        return self.root / "train" / f"{self.name}-{IMAGE_DETECTOR}"

    def det_path(self, detector: str | None = None) -> Path:
        return self.root / "train" / f"{self.name}-{detector or self.detector}" / "det" / "det.txt"

    def frame_path(self, frame: int) -> Path:
        return self.dir / self.info.img_dir / f"{frame:06d}{self.info.img_ext}"

    def read_frame(self, frame: int) -> np.ndarray:
        img = cv2.imread(str(self.frame_path(frame)))
        if img is None:
            raise FileNotFoundError(self.frame_path(frame))
        return img

    def frames(self, split: str = "full") -> Iterator[tuple[int, np.ndarray]]:
        first, last = frame_range(split, self.info.length)
        for frame in range(first, last + 1):
            yield frame, self.read_frame(frame)

    def gt(self, split: str = "full") -> tuple[np.ndarray, np.ndarray]:
        """(evaluated pedestrians, distractors) restricted to the split."""
        first, last = frame_range(split, self.info.length)
        pedestrians, distractors = split_gt(load_mot_txt(self.dir / "gt" / "gt.txt"))
        return select_frames(pedestrians, first, last), select_frames(distractors, first, last)

    def public_detections(self, split: str = "full", detector: str | None = None) -> np.ndarray:
        first, last = frame_range(split, self.info.length)
        return select_frames(load_mot_txt(self.det_path(detector)), first, last)


def list_sequences(root: Path = MOT17_DIR) -> list[str]:
    """Training sequences that are present on disk."""
    return [s for s in TRAIN_SEQUENCES if (root / "train" / f"{s}-{IMAGE_DETECTOR}").is_dir()]
