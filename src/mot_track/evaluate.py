"""Score tracker output with TrackEval (HOTA, CLEAR MOT, Identity).

TrackEval expects the MOTChallenge folder layout. To evaluate only one half of each sequence,
the ground truth and the tracker output are copied into a scratch folder with frames renumbered
from 1 and `seqLength` set to the half's length, so frames outside the split count for nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .config import MOT17_DIR, OUTPUTS_DIR
from .dataset import Sequence
from .mot_format import load_mot_txt
from .splits import frame_range, select_frames

SUMMARY_KEYS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "MT", "ML", "Frag")


def _import_trackeval():
    # TrackEval still uses the numpy aliases removed in numpy 1.24.
    for alias, typ in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, typ)
    import trackeval

    return trackeval


def _write_rows(path: Path, rows: np.ndarray, offset: int, n_cols: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            vals = [int(r[0]) - offset, int(r[1]), *r[2:n_cols]]
            f.write(",".join(f"{v:g}" if isinstance(v, float) else str(v) for v in vals) + "\n")


def prepare_split(
    work: Path, seqs: list[str], split: str, tracker_dirs: dict[str, Path], root: Path = MOT17_DIR
) -> None:
    """Write the renumbered ground truth and tracker files for one split into `work`."""
    if work.exists():
        shutil.rmtree(work)
    for name in seqs:
        seq = Sequence(name, root=root)
        first, last = frame_range(split, seq.info.length)
        gt_dir = work / "gt" / name
        gt = select_frames(load_mot_txt(seq.dir / "gt" / "gt.txt"), first, last)
        _write_rows(gt_dir / "gt" / "gt.txt", gt, first - 1, 9)
        (gt_dir / "seqinfo.ini").write_text(
            f"[Sequence]\nname={name}\nimDir=img1\nframeRate={seq.info.fps:g}\n"
            f"seqLength={last - first + 1}\nimWidth={seq.info.width}\n"
            f"imHeight={seq.info.height}\nimExt={seq.info.img_ext}\n"
        )
        for tracker, src in tracker_dirs.items():
            rows = select_frames(load_mot_txt(src / f"{name}.txt"), first, last)
            _write_rows(work / "trackers" / tracker / "data" / f"{name}.txt", rows, first - 1, 7)
    (work / "seqmap.txt").write_text("name\n" + "\n".join(seqs) + "\n")


def run_trackeval(work: Path, trackers: list[str], seqs: list[str]) -> dict:
    """Run TrackEval on a folder made by `prepare_split`; returns {tracker: {seq: metrics}}."""
    trackeval = _import_trackeval()
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(
        USE_PARALLEL=False,
        PRINT_RESULTS=False,
        PRINT_CONFIG=False,
        PRINT_ONLY_COMBINED=True,
        OUTPUT_SUMMARY=False,
        OUTPUT_DETAILED=False,
        PLOT_CURVES=False,
        TIME_PROGRESS=False,
        DISPLAY_LESS_PROGRESS=True,
    )
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        GT_FOLDER=str(work / "gt"),
        TRACKERS_FOLDER=str(work / "trackers"),
        BENCHMARK="MOT17",
        SPLIT_TO_EVAL="train",
        TRACKERS_TO_EVAL=trackers,
        SEQMAP_FILE=str(work / "seqmap.txt"),
        SKIP_SPLIT_FOL=True,
        PRINT_CONFIG=False,
    )
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator = trackeval.Evaluator(eval_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    res, _ = evaluator.evaluate([dataset], [m for m in metrics])
    out = {}
    for tracker in trackers:
        per_seq = res["MotChallenge2DBox"][tracker]
        out[tracker] = {seq: summarise(r["pedestrian"]) for seq, r in per_seq.items()}
    return out


def summarise(r: dict) -> dict:
    """Flatten TrackEval's nested result into the numbers reported here (percent, counts)."""
    hota, clear, ident = r["HOTA"], r["CLEAR"], r["Identity"]
    return {
        "HOTA": 100 * float(np.mean(hota["HOTA"])),
        "DetA": 100 * float(np.mean(hota["DetA"])),
        "AssA": 100 * float(np.mean(hota["AssA"])),
        "MOTA": 100 * float(clear["MOTA"]),
        "IDF1": 100 * float(ident["IDF1"]),
        "IDSW": int(clear["IDSW"]),
        "FP": int(clear["CLR_FP"]),
        "FN": int(clear["CLR_FN"]),
        "MT": int(clear["MT"]),
        "ML": int(clear["ML"]),
        "Frag": int(clear["Frag"]),
    }


def evaluate(tracker_dirs: dict[str, Path], seqs: list[str], split: str, tag: str) -> dict:
    """Evaluate result folders (`{name: dir with <seq>.txt}`) on one split."""
    work = OUTPUTS_DIR / "trackeval" / tag
    prepare_split(work, seqs, split, tracker_dirs)
    return run_trackeval(work, list(tracker_dirs), seqs)
