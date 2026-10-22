"""SORT and ByteTrack on top of the same Kalman filter and Hungarian matcher.

Both trackers share the motion model, so differences in the results come from the association
logic: SORT discards detections below a score threshold and matches once; ByteTrack keeps the
low-score boxes and uses them in a second matching round for tracks the first round missed
(typically people who are partly occluded and get a low detector score).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .association import fuse_score, iou_cost, linear_assignment
from .kalman import KalmanBoxFilter, xyah_to_xywh, xywh_to_xyah


class State(Enum):
    TENTATIVE = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class Track:
    """One target: Kalman state plus bookkeeping."""

    def __init__(self, kf: KalmanBoxFilter, box: np.ndarray, score: float, frame: int):
        self.kf = kf
        self.mean, self.cov = kf.initiate(xywh_to_xyah(box))
        self.score = float(score)
        self.id = 0
        self.state = State.TENTATIVE
        self.hits = 1
        self.start_frame = frame
        self.last_frame = frame

    @property
    def box(self) -> np.ndarray:
        """Current `xywh` estimate."""
        return xyah_to_xywh(self.mean)

    def predict(self) -> None:
        mean = self.mean.copy()
        if self.state != State.TRACKED:
            mean[7] = 0.0  # don't let the height velocity run away while the target is unseen
        self.mean, self.cov = self.kf.predict(mean, self.cov)

    def update(self, box: np.ndarray, score: float, frame: int) -> None:
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xywh_to_xyah(box))
        self.score = float(score)
        self.hits += 1
        self.last_frame = frame


def _boxes(tracks: list[Track]) -> np.ndarray:
    return np.array([t.box for t in tracks]).reshape(-1, 4)


def keep_output_box(box: np.ndarray, min_area: float, max_aspect: float) -> bool:
    """Drop tiny boxes and boxes much wider than a standing person (w / h > max_aspect)."""
    w, h = box[2], box[3]
    return w * h > min_area and w / max(h, 1e-6) <= max_aspect


class _Tracker:
    def __init__(self, min_area: float = 10.0, max_aspect: float = 1.6):
        self.kf = KalmanBoxFilter()
        self.frame = 0
        self.next_id = 1
        self.min_area = min_area
        self.max_aspect = max_aspect

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def _rows(self, tracks: list[Track]) -> np.ndarray:
        """Output rows (id, x, y, w, h, score) for the given tracks."""
        rows = [
            [t.id, *t.box, t.score]
            for t in tracks
            if keep_output_box(t.box, self.min_area, self.max_aspect)
        ]
        return np.array(rows, dtype=float).reshape(-1, 6)

    def update(self, dets: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SortConfig:
    det_thresh: float = 0.5  # detections below this are discarded
    iou_thresh: float = 0.3  # minimum IoU for a match
    max_age: int = 1  # frames a track survives without a match (1 in the SORT paper)
    min_hits: int = 3  # matches before a track is reported


class Sort(_Tracker):
    """SORT (Bewley et al., 2016): Kalman prediction + one round of IoU Hungarian matching."""

    def __init__(self, cfg: SortConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg or SortConfig()
        self.tracks: list[Track] = []

    def update(self, dets: np.ndarray) -> np.ndarray:
        """`dets` rows are (x, y, w, h, score); returns rows (id, x, y, w, h, score)."""
        self.frame += 1
        cfg = self.cfg
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        dets = dets[dets[:, 4] >= cfg.det_thresh]
        for t in self.tracks:
            t.predict()

        cost = iou_cost(_boxes(self.tracks), dets[:, :4])
        matches, _, unmatched_d = linear_assignment(cost, 1.0 - cfg.iou_thresh)
        for ti, di in matches:
            track = self.tracks[ti]
            track.update(dets[di, :4], dets[di, 4], self.frame)
            if track.state == State.TENTATIVE and track.hits >= cfg.min_hits:
                track.state = State.TRACKED
                track.id = self._new_id()
        for di in unmatched_d:
            self.tracks.append(Track(self.kf, dets[di, :4], dets[di, 4], self.frame))

        self.tracks = [t for t in self.tracks if self.frame - t.last_frame < cfg.max_age + 1]
        # SORT reports young tracks during the first frames so the sequence start isn't empty.
        for t in self.tracks:
            if t.state == State.TENTATIVE and self.frame <= cfg.min_hits:
                t.state = State.TRACKED
                t.id = self._new_id()
        shown = [t for t in self.tracks if t.state == State.TRACKED and t.last_frame == self.frame]
        return self._rows(shown)


@dataclass
class ByteConfig:
    track_thresh: float = 0.6  # high / low detection split
    low_thresh: float = 0.1  # boxes below this are ignored completely
    new_track_thresh: float = 0.7  # a high box must beat this to start a track
    match_thresh: float = 0.8  # max cost (1 - fused IoU) in the first round
    low_match_thresh: float = 0.5  # max cost (1 - IoU) in the second round
    unconfirmed_thresh: float = 0.7  # max cost for one-frame-old tracks
    track_buffer: int = 30  # frames a lost track is kept, at 30 FPS
    fuse: bool = True  # multiply IoU by the detection score in the first round


class ByteTrack(_Tracker):
    """ByteTrack (Zhang et al., 2022): associate every detection box, not just confident ones."""

    def __init__(self, cfg: ByteConfig | None = None, frame_rate: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg or ByteConfig()
        self.max_lost = int(round(frame_rate / 30.0 * self.cfg.track_buffer))
        self.tracked: list[Track] = []  # confirmed and tentative, seen last frame
        self.lost: list[Track] = []

    def _activate(self, track: Track) -> None:
        track.state = State.TRACKED
        if track.id == 0:
            track.id = self._new_id()

    def update(self, dets: np.ndarray) -> np.ndarray:
        """`dets` rows are (x, y, w, h, score); returns rows (id, x, y, w, h, score)."""
        self.frame += 1
        cfg = self.cfg
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        high = dets[dets[:, 4] >= cfg.track_thresh]
        low = dets[(dets[:, 4] > cfg.low_thresh) & (dets[:, 4] < cfg.track_thresh)]

        confirmed = [t for t in self.tracked if t.state == State.TRACKED]
        unconfirmed = [t for t in self.tracked if t.state == State.TENTATIVE]
        pool = confirmed + self.lost
        for t in pool + unconfirmed:
            t.predict()

        # Round 1: high-score boxes against confirmed and lost tracks.
        cost = iou_cost(_boxes(pool), high[:, :4])
        if cfg.fuse:
            cost = fuse_score(cost, high[:, 4])
        matches, rest_t, rest_h = linear_assignment(cost, cfg.match_thresh)
        active: list[Track] = []
        for ti, di in matches:
            pool[ti].update(high[di, :4], high[di, 4], self.frame)
            self._activate(pool[ti])
            active.append(pool[ti])

        # Round 2: low-score boxes against confirmed tracks still unmatched (not lost ones).
        second = [pool[i] for i in rest_t if pool[i].state == State.TRACKED]
        cost = iou_cost(_boxes(second), low[:, :4])
        matches, rest_s, _ = linear_assignment(cost, cfg.low_match_thresh)
        for ti, di in matches:
            second[ti].update(low[di, :4], low[di, 4], self.frame)
            active.append(second[ti])
        newly_lost = [second[i] for i in rest_s]
        for t in newly_lost:
            t.state = State.LOST

        # Tracks born last frame get one chance with the leftover high boxes.
        high = high[rest_h]
        cost = iou_cost(_boxes(unconfirmed), high[:, :4])
        if cfg.fuse:
            cost = fuse_score(cost, high[:, 4])
        matches, _, rest_h = linear_assignment(cost, cfg.unconfirmed_thresh)
        for ti, di in matches:
            unconfirmed[ti].update(high[di, :4], high[di, 4], self.frame)
            self._activate(unconfirmed[ti])
            active.append(unconfirmed[ti])

        # Remaining confident boxes start new tracks (confirmed straight away on frame 1).
        for di in rest_h:
            if high[di, 4] < cfg.new_track_thresh:
                continue
            track = Track(self.kf, high[di, :4], high[di, 4], self.frame)
            if self.frame == 1:
                self._activate(track)
            active.append(track)

        active_ids = {id(t) for t in active}
        lost = [t for t in self.lost if id(t) not in active_ids] + newly_lost
        lost = [t for t in lost if self.frame - t.last_frame <= self.max_lost]
        self.tracked, self.lost = self._remove_duplicates(active, lost)
        return self._rows([t for t in self.tracked if t.state == State.TRACKED])

    @staticmethod
    def _remove_duplicates(tracked: list[Track], lost: list[Track]):
        """A lost track that overlaps a live one (IoU > 0.85) is a duplicate: keep the older."""
        cost = iou_cost(_boxes(tracked), _boxes(lost))
        drop_t, drop_l = set(), set()
        for i, j in zip(*np.nonzero(cost < 0.15), strict=True):
            age_t = tracked[i].last_frame - tracked[i].start_frame
            age_l = lost[j].last_frame - lost[j].start_frame
            if age_t > age_l:
                drop_l.add(j)
            else:
                drop_t.add(i)
        return (
            [t for i, t in enumerate(tracked) if i not in drop_t],
            [t for j, t in enumerate(lost) if j not in drop_l],
        )


def make_tracker(name: str, frame_rate: float = 30.0, **params) -> _Tracker:
    if name == "sort":
        return Sort(SortConfig(**params))
    if name == "bytetrack":
        return ByteTrack(ByteConfig(**params), frame_rate=frame_rate)
    raise ValueError(f"unknown tracker {name!r}; use 'sort' or 'bytetrack'")
