"""Danger-zone alerts per identity, the use case from the construction-safety project.

Without tracking, a zone monitor can only say "someone is in the zone in this frame". With
persistent IDs it can raise one alert per person entering and staying, which is what a site
supervisor wants. The catch is that every ID switch inside the zone looks like a new person and
fires a duplicate alert. This module counts alerts from ground truth and from tracker output on
the static-camera sequences, so the effect of ID switches on alert counts can be measured.

    mot-zones    # -> reports/zone_alerts.{json,md}
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .analysis import clear_match
from .config import CAMERA, OUTPUTS_DIR, REPORTS_DIR
from .dataset import Sequence, list_sequences
from .mot_format import load_mot_txt

# Zone as fractions of the image (x0, y0, x1, y1): the lower middle of the frame, where people
# walk on the ground plane. Arbitrary, but fixed before looking at the results.
ZONE = (0.3, 0.55, 0.7, 1.0)
MIN_DWELL_S = 1.0  # a person must stay this long before an alert fires
GRACE_S = 0.5  # leaving for less than this does not end the visit


def feet_in_zone(rows: np.ndarray, zone_px: tuple[float, float, float, float]) -> np.ndarray:
    """Whether the bottom-centre point of each MOT row lies inside the zone."""
    x0, y0, x1, y1 = zone_px
    fx = rows[:, 2] + rows[:, 4] / 2
    fy = rows[:, 3] + rows[:, 5]
    return (fx >= x0) & (fx <= x1) & (fy >= y0) & (fy <= y1)


def zone_alerts(
    rows: np.ndarray, zone_px: tuple, fps: float, min_dwell_s=MIN_DWELL_S, grace_s=GRACE_S
) -> list[tuple[int, int]]:
    """Alerts as (identity, frame) pairs: one per visit that lasts at least `min_dwell_s`."""
    inside = rows[feet_in_zone(rows, zone_px)]
    min_frames = int(round(min_dwell_s * fps))
    grace = int(round(grace_s * fps))
    alerts = []
    for tid in np.unique(inside[:, 1]).astype(int):
        frames = np.sort(inside[inside[:, 1] == tid, 0].astype(int))
        visit_start, prev, fired = frames[0], frames[0], False
        for f in frames:
            if f - prev > grace + 1:  # the visit ended; a new one starts here
                visit_start, fired = f, False
            prev = f
            if not fired and f - visit_start + 1 >= min_frames:
                alerts.append((tid, int(f)))
                fired = True
    return sorted(alerts, key=lambda a: a[1])


def score_alerts(
    gt: np.ndarray, trk: np.ndarray, distractors: np.ndarray, zone: tuple, fps: float
) -> dict[str, int]:
    """Compare tracker alerts with ground-truth alerts, person by person.

    Each tracker alert is assigned to the ground-truth person its track covers in that frame
    (CLEAR MOT match). Per person, alerts up to the true number of visits are correct, extra
    ones are duplicates; alerts on no real person are false and visits without one are missed.
    """
    true_alerts = zone_alerts(gt, zone, fps)
    m = clear_match(gt, trk, distractors)
    owner = {
        (int(f), int(t)): int(g)
        for f, t, g in zip(gt[:, 0], m.track_id, gt[:, 1], strict=True)
        if t >= 0
    }
    per_gt: dict[int, int] = {}
    false = 0
    for tid, frame in zone_alerts(trk, zone, fps):
        gid = owner.get((frame, tid))
        if gid is None:
            false += 1
        else:
            per_gt[gid] = per_gt.get(gid, 0) + 1
    want: dict[int, int] = {}
    for gid, _ in true_alerts:
        want[gid] = want.get(gid, 0) + 1
    ids = set(want) | set(per_gt)
    return {
        "visits": len(true_alerts),
        "alerts": sum(per_gt.values()) + false,
        "correct": sum(min(want.get(i, 0), per_gt.get(i, 0)) for i in ids),
        "duplicate": sum(max(0, per_gt.get(i, 0) - want.get(i, 0)) for i in ids),
        "false": false,
        "missed": sum(max(0, want.get(i, 0) - per_gt.get(i, 0)) for i in ids),
    }


def zone_pixels(seq: Sequence, zone=ZONE) -> tuple[float, float, float, float]:
    w, h = seq.info.width, seq.info.height
    return zone[0] * w, zone[1] * h, zone[2] * w, zone[3] * h


def evaluate(seqs: list[str], configs: tuple[str, ...]) -> dict:
    out = {}
    for name in seqs:
        seq = Sequence(name)
        zone = zone_pixels(seq)
        gt, distractors = seq.gt("val")
        out[name] = {}
        for c in configs:
            trk = load_mot_txt(OUTPUTS_DIR / "results" / "val" / c / f"{name}.txt")[:, :7]
            out[name][c] = score_alerts(gt, trk, distractors, zone, seq.info.fps)
    return out


FIELDS = ("visits", "alerts", "correct", "duplicate", "false", "missed")


def to_markdown(res: dict) -> str:
    configs = list(next(iter(res.values())))
    lines = [
        "# Danger-zone alerts per identity (static-camera sequences, validation halves)",
        "",
        f"Zone: image fractions x {ZONE[0]}-{ZONE[2]}, y {ZONE[1]}-{ZONE[3]} (lower middle). "
        f"An alert fires once per visit when a person's feet stay in the zone for {MIN_DWELL_S:g} "
        f"s; gaps under {GRACE_S:g} s do not end a visit. `visits` is the number of alerts the "
        "ground truth produces. Tracker alerts are assigned to the person their track covers: "
        "`duplicate` alerts are the same person alerted again after an ID switch, `false` "
        "alerts come from a track not matched to an annotated pedestrian in that frame (a false "
        "track, a distractor such as a static person, or a box off by more than IoU 0.5), and "
        "`missed` visits got no alert. With under 30 visits per tracker, differences of a few "
        "alerts are within noise.",
        "",
        "| tracker | sequence | " + " | ".join(FIELDS) + " |",
        "|---|---|" + "---|" * len(FIELDS),
    ]
    for c in configs:
        total = dict.fromkeys(FIELDS, 0)
        for name, r in res.items():
            lines.append(f"| {c} | {name} | " + " | ".join(str(r[c][k]) for k in FIELDS) + " |")
            for k in FIELDS:
                total[k] += r[c][k]
        lines.append(f"| {c} | total | " + " | ".join(str(total[k]) for k in FIELDS) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="*", default=["SORT, tuned", "ByteTrack, tuned"])
    args = parser.parse_args(argv)
    seqs = [s for s in list_sequences() if CAMERA[s] == "static"]
    res = evaluate(seqs, tuple(args.configs))
    (REPORTS_DIR / "zone_alerts.json").write_text(json.dumps(res, indent=1))
    (REPORTS_DIR / "zone_alerts.md").write_text(to_markdown(res))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
