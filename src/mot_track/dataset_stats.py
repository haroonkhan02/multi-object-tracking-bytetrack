"""Per-sequence dataset statistics (`mot-check`): frames, density, occlusion, track lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .config import CAMERA, REPORTS_DIR
from .dataset import Sequence, list_sequences
from .mot_format import by_frame
from .viz import draw_tracks


def sequence_stats(seq: Sequence) -> dict:
    """Statistics of one sequence's evaluated ground truth (pedestrians only)."""
    gt, distractors = seq.gt("full")
    ids, counts = np.unique(gt[:, 1], return_counts=True)
    per_frame = np.array([len(v) for v in by_frame(gt).values()]) if len(gt) else np.zeros(1)
    vis = gt[:, 8]
    return {
        "sequence": seq.name,
        "camera": CAMERA.get(seq.name, "unknown"),
        "frames": seq.info.length,
        "fps": seq.info.fps,
        "resolution": f"{seq.info.width}x{seq.info.height}",
        "identities": int(len(ids)),
        "gt_boxes": int(len(gt)),
        "distractor_boxes": int(len(distractors)),
        "boxes_per_frame": round(float(per_frame.mean()), 2),
        "max_boxes_per_frame": int(per_frame.max()),
        "median_box_height_px": round(float(np.median(gt[:, 5])), 1) if len(gt) else 0.0,
        "mean_track_length": round(float(counts.mean()), 1) if len(ids) else 0.0,
        "occluded_fraction": round(float((vis < 0.5).mean()), 3) if len(gt) else 0.0,
        "public_detections": int(len(seq.public_detections("full"))),
    }


def markdown_table(rows: list[dict]) -> str:
    cols = [
        ("sequence", "Sequence"),
        ("camera", "Camera"),
        ("resolution", "Resolution"),
        ("fps", "FPS"),
        ("frames", "Frames"),
        ("identities", "IDs"),
        ("boxes_per_frame", "Boxes/frame"),
        ("median_box_height_px", "Median height (px)"),
        ("mean_track_length", "Mean track length"),
        ("occluded_fraction", "Visibility < 0.5"),
    ]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[k]) for k, _ in cols) + " |" for r in rows]
    return "\n".join(lines)


def save_sample(seq: Sequence, out: Path, frame: int | None = None) -> None:
    """Ground-truth boxes with identities on one frame, as a quick visual check."""
    frame = frame or seq.info.length // 2
    gt, _ = seq.gt("full")
    rows = by_frame(gt).get(frame, np.zeros((0, 9)))
    out.parent.mkdir(parents=True, exist_ok=True)
    img = draw_tracks(seq.read_frame(frame), rows)
    scale = 1280 / img.shape[1]
    cv2.imwrite(str(out), cv2.resize(img, None, fx=scale, fy=scale), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR)
    ap.add_argument("--sample", default="MOT17-04", help="sequence for the ground-truth figure")
    args = ap.parse_args(argv)

    names = list_sequences()
    if not names:
        raise SystemExit("no MOT17 sequences found; run `mot-download` first")
    rows = [sequence_stats(Sequence(n)) for n in names]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_stats.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out / "dataset_stats.md").write_text(markdown_table(rows) + "\n")
    print(markdown_table(rows))
    if args.sample in names:
        save_sample(Sequence(args.sample), args.out / "figures" / "gt_sample.jpg")


if __name__ == "__main__":
    main()
