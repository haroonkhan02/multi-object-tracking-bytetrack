"""Side-by-side demo video: two trackers on the same frames, with ID trails and a switch counter.

    mot-demo --seq MOT17-02 --start 331 --frames 150   # -> outputs/demo/*.mp4, assets/demo.gif

The counter is the running number of ID switches against the ground truth (CLEAR MOT rule), so
the video shows the same quantity the tables report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from .analysis import clear_match
from .config import OUTPUTS_DIR, ROOT
from .dataset import Sequence
from .mot_format import by_frame, load_mot_txt
from .viz import draw_tracks, id_color

CONFIGS = ("SORT, tuned", "ByteTrack, tuned")


def switch_frames(seq: Sequence, rows: np.ndarray, split: str = "val") -> np.ndarray:
    """Frame numbers of every ID switch of `rows` against the ground truth."""
    gt, distractors = seq.gt(split)
    m = clear_match(gt, rows, distractors)
    return np.sort(gt[m.switch, 0])


def draw_trails(image: np.ndarray, trails: dict[int, deque], thickness: int = 2) -> None:
    for tid, pts in trails.items():
        if len(pts) > 1:
            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts_arr], False, id_color(tid), thickness, cv2.LINE_AA)


def banner(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )


def render(
    seq: Sequence,
    results: dict[str, np.ndarray],
    start: int,
    n_frames: int,
    panel_width: int = 640,
    trail: int = 20,
) -> list[np.ndarray]:
    """Return composited frames (BGR), one panel per tracker side by side."""
    per_frame = {name: by_frame(rows) for name, rows in results.items()}
    switches = {name: switch_frames(seq, rows) for name, rows in results.items()}
    trails = {name: defaultdict(lambda: deque(maxlen=trail)) for name in results}
    scale = panel_width / seq.info.width
    size = (panel_width, int(round(seq.info.height * scale)))
    frames = []
    for frame in range(start, min(start + n_frames, seq.info.length + 1)):
        image = seq.read_frame(frame)
        panels = []
        for name in results:
            rows = per_frame[name].get(frame, np.zeros((0, 7)))
            for r in rows:
                trails[name][int(r[1])].append((r[2] + r[4] / 2, r[3] + r[5]))
            panel = draw_tracks(image, rows, thickness=3)
            draw_trails(panel, trails[name], thickness=3)
            panel = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
            n_sw = int(((switches[name] >= start) & (switches[name] <= frame)).sum())
            banner(panel, f"{name.split(',')[0]}   ID switches: {n_sw}")
            panels.append(panel)
        frames.append(np.hstack(panels))
    return frames


def write_mp4(frames: list[np.ndarray], path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def write_gif(frames: list[np.ndarray], path, fps: float, step: int = 3, width: int = 720) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for f in frames[::step]:
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=128))
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=int(1000 * step / fps), loop=0
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq", default="MOT17-02")
    parser.add_argument("--start", type=int, default=331, help="first frame (in the val half)")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--configs", nargs=2, default=list(CONFIGS))
    parser.add_argument("--gif", default=str(ROOT / "assets" / "demo.gif"))
    args = parser.parse_args(argv)

    seq = Sequence(args.seq)
    results = {
        c: load_mot_txt(OUTPUTS_DIR / "results" / "val" / c / f"{args.seq}.txt")[:, :7]
        for c in args.configs
    }
    frames = render(seq, results, args.start, args.frames)
    mp4 = OUTPUTS_DIR / "demo" / f"{args.seq}_{args.start}.mp4"
    write_mp4(frames, mp4, seq.info.fps)
    write_gif(frames, Path(args.gif), seq.info.fps)
    print(f"{len(frames)} frames -> {mp4} and {args.gif}")


if __name__ == "__main__":
    main()
