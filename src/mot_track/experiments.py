"""Tune tracker parameters on the first halves, then compare trackers on the second halves.

    mot-experiments tune   # grid search on the tune halves -> reports/tuning.{json,md}
    mot-experiments val    # defaults vs tuned, YOLO vs public detections -> reports/results_val.*
"""

from __future__ import annotations

import argparse
import itertools
import json

from .config import CAMERA, OUTPUTS_DIR, REPORTS_DIR
from .dataset import list_sequences
from .evaluate import SUMMARY_KEYS, evaluate
from .track import run, tracker_fps

DETS = "yolo11x"

# The first grids put the best values on their low edges, so the low end was extended twice.
SORT_GRID = {"det_thresh": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
             "max_age": [1, 10, 30], "min_hits": [1, 3]}  # fmt: skip
BYTE_GRID = {"track_thresh": [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7],
             "match_thresh": [0.7, 0.8, 0.9, 0.95],
             "track_buffer": [30, 60]}  # fmt: skip


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*space.values())]


def run_name(prefix: str, params: dict) -> str:
    return prefix + "-" + "-".join(str(v) for v in params.values())


def byte_params(p: dict) -> dict:
    # A new track needs a box a little above the high/low split, as in the ByteTrack code.
    return {**p, "new_track_thresh": round(p["track_thresh"] + 0.1, 2)}


def run_configs(configs: dict[str, tuple[str, str, dict]], split: str, tag: str) -> dict:
    """configs: {name: (tracker, detections, params)} -> {name: {metrics, params, fps}}."""
    seqs = list_sequences()
    dirs, fps = {}, {}
    for name, (tracker, dets, params) in configs.items():
        out = OUTPUTS_DIR / "results" / tag / name
        fps[name] = tracker_fps(run(tracker, dets, split, out, seqs, **params))
        dirs[name] = out
    scores = evaluate(dirs, seqs, split, tag)
    return {
        name: {
            "tracker": configs[name][0],
            "dets": configs[name][1],
            "params": configs[name][2],
            "tracker_fps": round(fps[name], 1),
            "combined": scores[name]["COMBINED_SEQ"],
            "per_seq": {s: m for s, m in scores[name].items() if s != "COMBINED_SEQ"},
        }
        for name in configs
    }


def fmt_row(name: str, m: dict, extra: list[str] | None = None) -> str:
    cells = [f"{m[k]:.1f}" if isinstance(m[k], float) else str(m[k]) for k in SUMMARY_KEYS]
    return "| " + " | ".join([name, *(extra or []), *cells]) + " |"


def header(first: list[str]) -> str:
    cols = [*first, *SUMMARY_KEYS]
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)


def tune() -> dict:
    configs = {}
    for p in grid(SORT_GRID):
        configs[run_name("sort", p)] = ("sort", DETS, p)
    for p in grid(BYTE_GRID):
        configs[run_name("byte", p)] = ("bytetrack", DETS, byte_params(p))
    results = run_configs(configs, "tune", "tune")
    best = {}
    for tracker in ("sort", "bytetrack"):
        mine = {k: v for k, v in results.items() if v["tracker"] == tracker}
        best[tracker] = max(mine, key=lambda k: mine[k]["combined"]["HOTA"])
    # Per-sequence numbers of every grid point would make the file huge; keep the totals.
    runs = {k: {f: v for f, v in r.items() if f != "per_seq"} for k, r in results.items()}
    payload = {"best": {t: results[k]["params"] for t, k in best.items()}, "runs": runs}
    (REPORTS_DIR / "tuning.json").write_text(json.dumps(payload, indent=1))

    lines = [
        "# Parameter search on the tune halves",
        "",
        f"Detections: {DETS} (COCO person class). Frames: first half of each MOT17 training "
        "sequence. Ranked by HOTA; the best row per tracker is used on the validation halves.",
        "",
    ]
    for tracker in ("sort", "bytetrack"):
        mine = sorted(
            (k for k in results if results[k]["tracker"] == tracker),
            key=lambda k: -results[k]["combined"]["HOTA"],
        )
        keys = list(SORT_GRID if tracker == "sort" else BYTE_GRID)
        lines += [f"## {tracker}", "", header(["run", *keys])]
        for k in mine:
            p = results[k]["params"]
            lines.append(fmt_row(k, results[k]["combined"], [str(p[x]) for x in keys]))
        lines.append("")
    (REPORTS_DIR / "tuning.md").write_text("\n".join(lines))
    print(f"best: {payload['best']}")
    return payload


def validate() -> dict:
    best = json.loads((REPORTS_DIR / "tuning.json").read_text())["best"]
    configs = {
        "SORT, paper defaults": ("sort", DETS, {}),
        "SORT, tuned": ("sort", DETS, best["sort"]),
        "ByteTrack, paper defaults": ("bytetrack", DETS, {}),
        "ByteTrack, tuned": ("bytetrack", DETS, best["bytetrack"]),
        "SORT, public FRCNN dets": ("sort", "public-FRCNN", {}),
        "ByteTrack, public FRCNN dets": ("bytetrack", "public-FRCNN", {}),
    }
    results = run_configs(configs, "val", "val")
    det_timing = json.loads((OUTPUTS_DIR / "detections" / DETS / "timing.json").read_text())
    det_fps = sum(t["frames"] for t in det_timing.values()) / sum(
        t["seconds"] for t in det_timing.values()
    )
    payload = {"detector": DETS, "detector_fps": round(det_fps, 2), "runs": results}
    (REPORTS_DIR / "results_val.json").write_text(json.dumps(payload, indent=1))

    lines = [
        "# Validation halves (second half of each MOT17 training sequence)",
        "",
        f"Detector: {DETS} at 1280 px on MPS, {det_fps:.1f} FPS. Tracker FPS excludes detection.",
        "",
        header(["config", "dets", "tracker FPS"]),
    ]
    for name, r in results.items():
        lines.append(fmt_row(name, r["combined"], [r["dets"], f"{r['tracker_fps']:.0f}"]))
    for name in ("SORT, tuned", "ByteTrack, tuned"):
        lines += ["", f"## Per sequence: {name}", "", header(["sequence", "camera"])]
        for seq, m in results[name]["per_seq"].items():
            lines.append(fmt_row(seq, m, [CAMERA[seq]]))
    (REPORTS_DIR / "results_val.md").write_text("\n".join(lines) + "\n")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["tune", "val"])
    args = parser.parse_args(argv)
    tune() if args.stage == "tune" else validate()


if __name__ == "__main__":
    main()
