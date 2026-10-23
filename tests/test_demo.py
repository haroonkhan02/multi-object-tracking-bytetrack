import numpy as np

from mot_track.dataset import Sequence
from mot_track.demo import render, switch_frames, write_gif


def results(seq):
    gt, _ = seq.gt()
    one = gt[gt[:, 1] == 1][:, :7].copy()
    swapped = one.copy()
    swapped[swapped[:, 0] >= 8, 1] = 5  # ID switch at frame 8
    return {"A": one, "B": swapped}


def test_switch_frames_counts_against_ground_truth(mot_root):
    seq = Sequence("MOT17-02", root=mot_root)
    res = results(seq)
    assert len(switch_frames(seq, res["A"])) == 0
    assert list(switch_frames(seq, res["B"])) == [8]


def test_render_side_by_side_and_gif(mot_root, tmp_path):
    seq = Sequence("MOT17-02", root=mot_root)
    frames = render(seq, results(seq), start=6, n_frames=4, panel_width=100)
    assert len(frames) == 4
    assert frames[0].shape == (50, 200, 3) and frames[0].dtype == np.uint8
    gif = tmp_path / "demo.gif"
    write_gif(frames, gif, fps=30, step=1, width=200)
    assert gif.stat().st_size > 0
