"""Run a COCO-pretrained YOLO person detector over MOT17 and cache the boxes.

Detection is the slow part of tracking-by-detection, so it runs once per sequence and the
trackers read the cache. Boxes are kept down to a low score (0.05): ByteTrack's second
association stage needs the low-confidence ones that SORT throws away.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import MODELS_DIR, OUTPUTS_DIR
from .dataset import Sequence, list_sequences

PERSON = 0  # COCO class id
DET_COLUMNS = ("frame", "x", "y", "w", "h", "score")


def pick_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def det_cache_path(source: str, seq: str) -> Path:
    return OUTPUTS_DIR / "detections" / source / f"{seq}.npy"


def load_detections(source: str, seq: str, split: str = "full") -> np.ndarray:
    """Cached detections as rows (frame, x, y, w, h, score).

    `source` is a cached YOLO model name or `public-<DPM|FRCNN|SDP>` for the MOT17 public sets.
    """
    if source.startswith("public-"):
        rows = Sequence(seq).public_detections(split, detector=source.split("-", 1)[1])
        return rows[:, [0, 2, 3, 4, 5, 6]] if len(rows) else np.zeros((0, 6))
    from .splits import frame_range, select_frames

    rows = np.load(det_cache_path(source, seq))
    first, last = frame_range(split, Sequence(seq).info.length)
    return select_frames(rows, first, last)


def boxes_from_result(result, frame: int) -> np.ndarray:
    """Convert one ultralytics result into rows (frame, x, y, w, h, score)."""
    xyxy = result.boxes.xyxy.cpu().numpy().reshape(-1, 4)
    conf = result.boxes.conf.cpu().numpy().reshape(-1)
    rows = np.empty((len(conf), 6), dtype=np.float32)
    rows[:, 0] = frame
    rows[:, 1:3] = xyxy[:, :2]
    rows[:, 3:5] = xyxy[:, 2:] - xyxy[:, :2]
    rows[:, 5] = conf
    return rows


def detect_sequence(model, seq: Sequence, imgsz: int, conf: float, device: str) -> tuple:
    rows, seconds = [], 0.0
    for frame, img in seq.frames():
        start = time.perf_counter()
        result = model.predict(
            img, imgsz=imgsz, conf=conf, classes=[PERSON], device=device, verbose=False
        )[0]
        seconds += time.perf_counter() - start
        rows.append(boxes_from_result(result, frame))
    return np.concatenate(rows) if rows else np.zeros((0, 6), np.float32), seconds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="yolo11x.pt", help="ultralytics weights name")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--force", action="store_true", help="re-run cached sequences")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    device = pick_device()
    model = YOLO(str(MODELS_DIR / args.model))
    source = Path(args.model).stem
    timing_path = OUTPUTS_DIR / "detections" / source / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    for name in args.seqs or list_sequences():
        out = det_cache_path(source, name)
        if out.exists() and not args.force:
            print(f"{name}: cached")
            continue
        seq = Sequence(name)
        rows, seconds = detect_sequence(model, seq, args.imgsz, args.conf, device)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, rows)
        timing[name] = {"frames": seq.info.length, "seconds": round(seconds, 2)}
        timing_path.write_text(json.dumps(timing, indent=2))
        print(f"{name}: {len(rows)} boxes, {seq.info.length / seconds:.1f} FPS on {device}")


if __name__ == "__main__":
    main()
