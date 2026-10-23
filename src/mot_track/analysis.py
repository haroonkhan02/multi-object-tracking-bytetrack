"""Error analysis on the validation halves: where do the trackers lose people and identities?

Every evaluated ground-truth box is matched to the tracker output frame by frame with the CLEAR
MOT rule TrackEval uses (IoU >= 0.5; last frame's pairs win ties; tracker boxes on distractors
are dropped first). Each box then carries attributes from the annotation, namely visibility,
height and its overlap with other people, so recall and ID switches can be split by difficulty.
An ID switch also records how many frames the person had been unmatched before it, which
separates swaps between neighbours (gap 0) from identities lost during an occlusion.

    mot-analyse    # -> reports/error_analysis.{json,md}, reports/figures/error_analysis.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boxes import iou_matrix, xywh_to_xyxy
from .config import CAMERA, OUTPUTS_DIR, REPORTS_DIR
from .dataset import Sequence, list_sequences
from .mot_format import by_frame, load_mot_txt

IOU_THRESH = 0.5
CONFIGS = ("SORT, tuned", "ByteTrack, tuned")
DETS = "yolo11x"

VIS_BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
HEIGHT_BINS = ((0, 50), (50, 100), (100, 200), (200, 10_000))
OVERLAP_BINS = ((-1.0, 1e-9), (1e-9, 0.1), (0.1, 0.3), (0.3, 1.01))
GAP_BINS = ((0, 0), (1, 10), (11, 30), (31, 10**9))


@dataclass
class Matched:
    """Per ground-truth row: matched track id (-1 if missed), ID-switch flag, gap before it."""

    track_id: np.ndarray
    switch: np.ndarray
    gap: np.ndarray


def _hungarian(score: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    if score.size == 0:
        return []
    rows, cols = linear_sum_assignment(-np.where(valid, score, 0.0))
    return [(r, c) for r, c in zip(rows, cols, strict=True) if valid[r, c]]


def drop_distractor_matches(trk: np.ndarray, gt: np.ndarray, distractors: np.ndarray) -> np.ndarray:
    """Remove tracker rows of one frame that match a distractor (as MOT17 evaluation does)."""
    if len(trk) == 0 or len(distractors) == 0:
        return trk
    every = np.concatenate([gt[:, 2:6], distractors[:, 2:6]])
    iou = iou_matrix(xywh_to_xyxy(every), xywh_to_xyxy(trk[:, 2:6]))
    pairs = _hungarian(iou, iou >= IOU_THRESH)
    drop = [c for r, c in pairs if r >= len(gt)]
    return np.delete(trk, drop, axis=0)


def clear_match(gt: np.ndarray, trk: np.ndarray, distractors: np.ndarray | None = None) -> Matched:
    """CLEAR MOT matching of MOT rows `gt` (pedestrians) and `trk` (tracker output)."""
    n = len(gt)
    out = Matched(np.full(n, -1), np.zeros(n, bool), np.full(n, -1))
    gt_idx = by_frame(np.c_[gt[:, :6], np.arange(n)]) if n else {}
    trk_frames = by_frame(trk)
    dis_frames = by_frame(distractors) if distractors is not None else {}
    prev_step: dict[int, int] = {}  # gt id -> track id in the previous frame
    last: dict[int, tuple[int, int]] = {}  # gt id -> (track id, frame) of the last match
    for frame in sorted(gt_idx):
        g = gt_idx[frame]
        t = trk_frames.get(frame, np.zeros((0, 7)))
        t = drop_distractor_matches(t, g, dis_frames.get(frame, np.zeros((0, 9))))
        iou = iou_matrix(xywh_to_xyxy(g[:, 2:6]), xywh_to_xyxy(t[:, 2:6]))
        cont = np.array(
            [[prev_step.get(int(gi)) == int(ti) for ti in t[:, 1]] for gi in g[:, 1]], dtype=bool
        )
        valid = iou >= IOU_THRESH
        score = iou + 1000.0 * (cont.reshape(iou.shape) & valid)
        prev_step = {}
        for r, c in _hungarian(score, valid):
            gid, tid, row = int(g[r, 1]), int(t[c, 1]), int(g[r, 6])
            out.track_id[row] = tid
            if gid in last and last[gid][0] != tid:
                out.switch[row] = True
                out.gap[row] = frame - last[gid][1] - 1
            last[gid] = (tid, frame)
            prev_step[gid] = tid
    return out


def max_overlap(gt: np.ndarray) -> np.ndarray:
    """For every ground-truth box, its largest IoU with another person in the same frame."""
    out = np.zeros(len(gt))
    for rows in by_frame(np.c_[gt[:, :6], np.arange(len(gt))]).values():
        if len(rows) < 2:
            continue
        iou = iou_matrix(xywh_to_xyxy(rows[:, 2:6]), xywh_to_xyxy(rows[:, 2:6]))
        np.fill_diagonal(iou, 0.0)
        out[rows[:, 6].astype(int)] = iou.max(axis=1)
    return out


def best_det_score(gt: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """Highest score among detections with IoU >= 0.5 to each ground-truth box (0 if none)."""
    out = np.zeros(len(gt))
    det_frames = by_frame(dets)
    for frame, rows in by_frame(np.c_[gt[:, :6], np.arange(len(gt))]).items():
        d = det_frames.get(frame)
        if d is None:
            continue
        iou = iou_matrix(xywh_to_xyxy(rows[:, 2:6]), xywh_to_xyxy(d[:, 1:5]))
        scores = np.where(iou >= IOU_THRESH, d[None, :, 5], 0.0)
        out[rows[:, 6].astype(int)] = scores.max(axis=1)
    return out


def in_bin(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (values >= lo) & (values < hi)


def breakdown(values: np.ndarray, bins, recall: dict, switches: dict) -> list[dict]:
    """Recall and ID switches per 1000 boxes for each tracker, per bin of `values`."""
    rows = []
    for lo, hi in bins:
        m = in_bin(values, lo, hi)
        row = {"lo": lo, "hi": hi, "boxes": int(m.sum())}
        for name in recall:
            k = max(int(m.sum()), 1)
            row[f"recall {name}"] = float(recall[name][m].mean()) if m.any() else 0.0
            if name in switches:
                row[f"IDSW/1k {name}"] = 1000.0 * float(switches[name][m].sum()) / k
        rows.append(row)
    return rows


def collect(seqs: list[str], configs=CONFIGS, dets: str = DETS) -> dict:
    """Match every config on the validation halves; returns stacked per-box arrays."""
    from .detect import load_detections

    cols: dict[str, list] = {"seq": [], "vis": [], "height": [], "overlap": [], "det": []}
    per = {c: {"matched": [], "switch": [], "gap": []} for c in configs}
    for name in seqs:
        gt, distractors = Sequence(name).gt("val")
        cols["seq"] += [name] * len(gt)
        cols["vis"].append(gt[:, 8])
        cols["height"].append(gt[:, 5])
        cols["overlap"].append(max_overlap(gt))
        cols["det"].append(best_det_score(gt, load_detections(dets, name, "val")))
        for c in configs:
            trk = load_mot_txt(OUTPUTS_DIR / "results" / "val" / c / f"{name}.txt")[:, :7]
            m = clear_match(gt, trk, distractors)
            per[c]["matched"].append(m.track_id >= 0)
            per[c]["switch"].append(m.switch)
            per[c]["gap"].append(m.gap)
    out = {k: (np.array(v) if k == "seq" else np.concatenate(v)) for k, v in cols.items()}
    out["trackers"] = {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in per.items()}
    return out


def _gap_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}")


def summarise(data: dict, det_thresholds: tuple[float, ...] = (0.2, 0.1)) -> dict:
    trk = data["trackers"]
    recall = {c: d["matched"] for c, d in trk.items()}
    for t in det_thresholds:
        recall[f"dets >= {t:g}"] = data["det"] >= t
    recall["any det"] = data["det"] > 0
    switches = {c: d["switch"] for c, d in trk.items()}
    gaps = {}
    for c, d in trk.items():
        g = d["gap"][d["switch"]]
        gaps[c] = {_gap_label(lo, hi): int(((g >= lo) & (g <= hi)).sum()) for lo, hi in GAP_BINS}
    seqs = []
    for s in sorted(set(data["seq"])):
        m = data["seq"] == s
        row = {"seq": s, "camera": CAMERA[s], "boxes": int(m.sum())}
        row["mean overlap"] = float(data["overlap"][m].mean())
        row["share vis < 0.4"] = float((data["vis"][m] < 0.4).mean())
        for c, d in trk.items():
            row[f"IDSW {c}"] = int(d["switch"][m].sum())
        seqs.append(row)
    return {
        "visibility": breakdown(data["vis"], VIS_BINS, recall, switches),
        "height": breakdown(data["height"], HEIGHT_BINS, recall, switches),
        "overlap": breakdown(data["overlap"], OVERLAP_BINS, recall, switches),
        "switch_gaps": gaps,
        "sequences": seqs,
        "totals": {
            c: {"recall": float(d["matched"].mean()), "IDSW": int(d["switch"].sum())}
            for c, d in trk.items()
        },
        "boxes": int(len(data["vis"])),
    }


def _bin_label(lo: float, hi: float, kind: str) -> str:
    if kind == "overlap":
        return {0: "none", 1: "0-0.1", 2: "0.1-0.3", 3: "> 0.3"}[
            [b[0] for b in OVERLAP_BINS].index(lo)
        ]
    if kind == "height":
        return f"{lo}-{hi} px" if hi < 10_000 else f">= {lo} px"
    return f"{lo:.1f}-{min(hi, 1.0):.1f}"


def to_markdown(s: dict) -> str:
    trackers = list(s["totals"])
    lines = [
        "# Error analysis on the validation halves",
        "",
        f"{s['boxes']} evaluated ground-truth boxes (second half of the seven MOT17 training "
        "sequences), matched to the tuned trackers with the CLEAR MOT rule at IoU 0.5. "
        "`dets >= t` is the share of boxes covered by a YOLO11x detection with score >= t "
        "(IoU >= 0.5), an upper bound on what a tracker using those boxes can recall. "
        "`IDSW/1k` is ID switches per 1000 boxes in the bin.",
        "",
        "| tracker | recall | ID switches |",
        "|---|---|---|",
    ]
    for c, t in s["totals"].items():
        lines.append(f"| {c} | {100 * t['recall']:.1f} % | {t['IDSW']} |")
    titles = {
        "visibility": "By visibility (annotated fraction of the person that is not occluded)",
        "height": "By box height",
        "overlap": "By overlap with another person (max IoU with any other box in the frame)",
    }
    for kind, title in titles.items():
        rows = s[kind]
        rec_keys = [k for k in rows[0] if k.startswith("recall ")]
        sw_keys = [k for k in rows[0] if k.startswith("IDSW/1k ")]
        cols = ["bin", "boxes", *rec_keys, *sw_keys]
        lines += ["", f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            cells = [_bin_label(r["lo"], r["hi"], kind), str(r["boxes"])]
            cells += [f"{100 * r[k]:.1f}" for k in rec_keys]
            cells += [f"{r[k]:.1f}" for k in sw_keys]
            lines.append("| " + " | ".join(cells) + " |")
    gap_cols = list(next(iter(s["switch_gaps"].values())))
    lines += [
        "",
        "## ID switches by frames unmatched before the switch",
        "",
        "Gap 0 means the person was matched in the previous frame and the ID jumped to another "
        "track (usually a swap between neighbours). Longer gaps are identities lost while the "
        "person was missed, typically during an occlusion.",
        "",
        "| tracker | " + " | ".join(f"gap {g}" for g in gap_cols) + " |",
        "|---|" + "---|" * len(gap_cols),
    ]
    for c, g in s["switch_gaps"].items():
        lines.append(f"| {c} | " + " | ".join(str(v) for v in g.values()) + " |")
    lines += [
        "",
        "## Per sequence",
        "",
        "| sequence | camera | boxes | mean overlap | share vis < 0.4 | "
        + " | ".join(f"IDSW {c}" for c in trackers)
        + " |",
        "|---|---|---|---|---|" + "---|" * len(trackers),
    ]
    for r in s["sequences"]:
        cells = [r["seq"], r["camera"], str(r["boxes"]), f"{r['mean overlap']:.3f}",
                 f"{100 * r['share vis < 0.4']:.0f} %"]  # fmt: skip
        cells += [str(r[f"IDSW {c}"]) for c in trackers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot(s: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    vis = s["visibility"]
    x = np.arange(len(vis))
    labels = [_bin_label(r["lo"], r["hi"], "visibility") for r in vis]
    rec_keys = [k for k in vis[0] if k.startswith("recall ")]
    styles = {"any det": ":", "dets >= 0.1": "--", "dets >= 0.2": "--"}
    for k in rec_keys:
        name = k.removeprefix("recall ")
        axes[0].plot(x, [100 * r[k] for r in vis], styles.get(name, "-"), marker="o", label=name)
    axes[0].set(xticks=x, xticklabels=labels, xlabel="visibility", ylabel="recall (%)",
                title="Recall by visibility")  # fmt: skip
    axes[0].legend(fontsize=8)

    trackers = list(s["totals"])
    width = 0.38
    for i, c in enumerate(trackers):
        axes[1].bar(x + (i - 0.5) * width, [r[f"IDSW/1k {c}"] for r in vis], width, label=c)
    axes[1].set(xticks=x, xticklabels=labels, xlabel="visibility",
                ylabel="ID switches per 1000 boxes", title="ID switches by visibility")  # fmt: skip
    axes[1].legend(fontsize=8)

    gaps = list(next(iter(s["switch_gaps"].values())))
    gx = np.arange(len(gaps))
    for i, c in enumerate(trackers):
        axes[2].bar(gx + (i - 0.5) * width, list(s["switch_gaps"][c].values()), width, label=c)
    axes[2].set(xticks=gx, xticklabels=gaps,
                xlabel="frames unmatched before the switch", ylabel="ID switches",
                title="Swaps vs. identities lost in occlusion")  # fmt: skip
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seqs", nargs="*", default=None)
    args = parser.parse_args(argv)
    s = summarise(collect(args.seqs or list_sequences()))
    (REPORTS_DIR / "error_analysis.json").write_text(json.dumps(s, indent=1))
    (REPORTS_DIR / "error_analysis.md").write_text(to_markdown(s))
    plot(s, REPORTS_DIR / "figures" / "error_analysis.png")
    for c, t in s["totals"].items():
        print(f"{c}: recall {100 * t['recall']:.1f} %, IDSW {t['IDSW']}")
