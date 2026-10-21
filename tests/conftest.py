"""Synthetic MOT17-style sequence so tests need no dataset."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

SEQ_LEN = 10


@pytest.fixture
def mot_root(tmp_path):
    """Two identities that walk across a 200x100 frame, plus a distractor and a conf-0 row."""
    seq_dir = tmp_path / "train" / "MOT17-02-FRCNN"
    (seq_dir / "img1").mkdir(parents=True)
    (seq_dir / "gt").mkdir()
    (seq_dir / "det").mkdir()
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate=30\n"
        f"seqLength={SEQ_LEN}\nimWidth=200\nimHeight=100\nimExt=.jpg\n"
    )
    gt, det = [], []
    for f in range(1, SEQ_LEN + 1):
        cv2.imwrite(str(seq_dir / "img1" / f"{f:06d}.jpg"), np.full((100, 200, 3), 40, np.uint8))
        gt.append(f"{f},1,{10 + 5 * f},20,20,40,1,1,1.0")
        gt.append(f"{f},2,{150 - 5 * f},30,20,40,1,1,0.3")
        det.append(f"{f},-1,{10 + 5 * f},20,20,40,0.9,-1,-1,-1")
    gt.append("1,3,5,5,10,10,1,7,1.0")  # static person: distractor
    gt.append("1,4,50,50,10,10,0,1,1.0")  # conf 0: ignored by the benchmark
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt) + "\n")
    (seq_dir / "det" / "det.txt").write_text("\n".join(det) + "\n")
    return tmp_path
