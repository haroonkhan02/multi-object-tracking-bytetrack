<h1 align="center">multi-object-tracking-bytetrack</h1>

<p align="center">
Tracking pedestrians with persistent IDs: YOLO detections, a Kalman-filter SORT baseline and ByteTrack, scored on MOT17 with HOTA, MOTA and IDF1.
</p>

<p align="center">
  <a href="https://github.com/haroonkhan02/multi-object-tracking-bytetrack/actions/workflows/ci.yml"><img src="https://github.com/haroonkhan02/multi-object-tracking-bytetrack/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
</p>

> **Work in progress.** Detection (COCO-pretrained YOLO11x, person class), the Kalman filter,
> SORT and ByteTrack are implemented and scored with TrackEval on the validation halves. First
> result: tuned ByteTrack reaches HOTA 51.1 and IDF1 59.8 against 48.9 and 56.4 for tuned SORT,
> with about half the ID switches ([full tables](reports/results_val.md)). Still to come: error
> analysis of occlusions and crowded scenes, a tracked demo video and the final write-up.

## Approach

```mermaid
flowchart LR
    V[Video frames] --> D[YOLO detector]
    D --> T["Tracker: Kalman filter + Hungarian matching"]
    T --> R[MOT result files]
    R --> E[TrackEval: HOTA, MOTA, IDF1]
```

- **Data:** the seven MOT17 training sequences. The test set has no public ground truth, so each
  sequence is split in half: the first half is for tuning tracker parameters, the second half for
  the reported numbers.
- **Detector:** a COCO-pretrained YOLO, keeping only the person class.
- **Trackers:** SORT as the baseline, then ByteTrack, which also matches low-confidence
  detections instead of discarding them. Both use a constant-velocity Kalman filter and
  Hungarian assignment.
- **Metrics:** HOTA, MOTA, IDF1, ID switches and FPS, computed with TrackEval.

## Getting started

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
mot-download     # 5.9 GB archive from motchallenge.net, resumable; extracts the train sequences
mot-check        # per-sequence statistics -> reports/dataset_stats.md
uv pip install -e ".[detect,eval]"
mot-detect                 # YOLO11x person boxes for every frame, cached in outputs/
mot-experiments tune       # parameter grid on the first halves -> reports/tuning.md
mot-experiments val        # SORT vs ByteTrack on the second halves -> reports/results_val.md
pytest -q
```

## Repository structure

```text
src/mot_track/
  config.py         paths and MOT17 constants
  download.py       resumable download and selective extraction
  mot_format.py     read/write MOTChallenge text files, ground-truth classes
  dataset.py        Sequence: frames, ground truth, public detections
  splits.py         tune / validation halves
  boxes.py          box conversions and IoU
  viz.py            draw boxes and IDs
  dataset_stats.py  per-sequence statistics and a ground-truth figure
  detect.py         YOLO person detector, cached per sequence
  kalman.py         constant-velocity Kalman filter on (cx, cy, aspect, height)
  association.py    IoU cost, score fusion, Hungarian matching
  trackers.py       SORT and ByteTrack
  track.py          run a tracker over cached detections
  evaluate.py       TrackEval wrapper (HOTA, CLEAR, Identity) on split halves
  experiments.py    parameter search and validation runs
tests/              synthetic sequence fixtures, no dataset needed
reports/            dataset statistics, tuning grid, validation results
```

## License

MIT, see [LICENSE](LICENSE). MOT17 is distributed by [motchallenge.net](https://motchallenge.net/data/MOT17/) under CC BY-NC-SA 3.0 and is not included in this repository.
