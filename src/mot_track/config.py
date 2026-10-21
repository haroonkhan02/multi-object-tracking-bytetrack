"""Paths and dataset constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MOT17_DIR = DATA_DIR / "MOT17"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

MOT17_URL = "https://motchallenge.net/data/MOT17.zip"
MOT17_ZIP_BYTES = 5_860_214_001

# The seven MOT17 training sequences (the test set has no public ground truth).
TRAIN_SEQUENCES = tuple(f"MOT17-{n:02d}" for n in (2, 4, 5, 9, 10, 11, 13))

# Each sequence ships three sets of public detections; the images are identical.
DETECTORS = ("DPM", "FRCNN", "SDP")
IMAGE_DETECTOR = "FRCNN"

# Camera motion per sequence, from the table in the MOT16 paper (Milan et al., 2016).
CAMERA = {
    "MOT17-02": "static",
    "MOT17-04": "static",
    "MOT17-05": "moving",
    "MOT17-09": "static",
    "MOT17-10": "moving",
    "MOT17-11": "moving",
    "MOT17-13": "moving",
}

# MOT17 ground-truth class ids (MOTChallenge format).
CLASS_PEDESTRIAN = 1
CLASS_DISTRACTORS = (2, 7, 8, 12)  # person on vehicle, static person, distractor, reflection
