"""Run a tracker over cached detections and write MOTChallenge result files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import OUTPUTS_DIR
from .dataset import Sequence, list_sequences
from .detect import load_detections
from .mot_format import by_frame, write_results
from .splits import frame_range
from .trackers import make_tracker


def track_sequence(
    seq: Sequence, dets: np.ndarray, tracker_name: str, split: str = "full", **params
) -> tuple[np.ndarray, float]:
    """Track one sequence. `dets` rows are (frame, x, y, w, h, score).

    Returns result rows (frame, id, x, y, w, h, score) and the tracker-only time in seconds.
    """
    tracker = make_tracker(tracker_name, frame_rate=seq.info.fps, **params)
    first, last = frame_range(split, seq.info.length)
    per_frame = by_frame(dets)
    empty = np.zeros((0, 5))
    rows, seconds = [], 0.0
    for frame in range(first, last + 1):
        frame_dets = per_frame.get(frame)
        frame_dets = empty if frame_dets is None else frame_dets[:, 1:6]
        start = time.perf_counter()
        out = tracker.update(frame_dets)
        seconds += time.perf_counter() - start
        if len(out):
            rows.append(np.c_[np.full(len(out), frame), out])
    result = np.concatenate(rows) if rows else np.zeros((0, 7))
    return result, seconds


def run(
    tracker_name: str,
    source: str,
    split: str,
    out_dir: Path,
    seqs: list[str] | None = None,
    **params,
) -> dict:
    """Track every sequence, write `<out_dir>/<seq>.txt`, return timing per sequence."""
    timing = {}
    for name in seqs or list_sequences():
        seq = Sequence(name)
        dets = load_detections(source, name, split)
        rows, seconds = track_sequence(seq, dets, tracker_name, split, **params)
        write_results(out_dir / f"{name}.txt", rows)
        first, last = frame_range(split, seq.info.length)
        timing[name] = {"frames": last - first + 1, "seconds": seconds}
    return timing


def tracker_fps(timing: dict) -> float:
    frames = sum(t["frames"] for t in timing.values())
    seconds = sum(t["seconds"] for t in timing.values())
    return frames / max(seconds, 1e-9)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", choices=["sort", "bytetrack"], default="bytetrack")
    parser.add_argument("--dets", default="yolo11x", help="yolo11x or public-FRCNN/SDP/DPM")
    parser.add_argument("--split", choices=["tune", "val", "full"], default="val")
    parser.add_argument("--params", default="{}", help='JSON, e.g. \'{"track_thresh": 0.5}\'')
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--name", default=None, help="result folder name")
    args = parser.parse_args(argv)

    name = args.name or f"{args.tracker}-{args.dets}-{args.split}"
    out_dir = OUTPUTS_DIR / "results" / name
    timing = run(args.tracker, args.dets, args.split, out_dir, args.seqs, **json.loads(args.params))
    print(f"{name}: tracker {tracker_fps(timing):.0f} FPS -> {out_dir}")


if __name__ == "__main__":
    main()
