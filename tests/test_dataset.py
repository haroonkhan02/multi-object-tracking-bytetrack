import numpy as np

from mot_track.dataset import Sequence, list_sequences
from mot_track.dataset_stats import markdown_table, save_sample, sequence_stats
from mot_track.viz import draw_tracks, id_color


def test_list_sequences(mot_root):
    assert list_sequences(mot_root) == ["MOT17-02"]


def test_sequence_access(mot_root):
    seq = Sequence("MOT17-02", root=mot_root)
    assert seq.read_frame(1).shape == (100, 200, 3)
    gt, dis = seq.gt("val")
    assert gt[:, 0].min() == 6 and len(dis) == 0  # the distractor only exists in frame 1
    assert len(seq.public_detections("tune")) == 5
    assert [f for f, _ in seq.frames("val")] == [6, 7, 8, 9, 10]


def test_sequence_stats(mot_root):
    stats = sequence_stats(Sequence("MOT17-02", root=mot_root))
    assert stats["identities"] == 2
    assert stats["gt_boxes"] == 20
    assert stats["distractor_boxes"] == 1
    assert stats["boxes_per_frame"] == 2.0
    assert stats["occluded_fraction"] == 0.5  # identity 2 has visibility 0.3 in every frame
    assert stats["camera"] == "static"
    assert "MOT17-02" in markdown_table([stats])


def test_save_sample_and_draw(mot_root, tmp_path):
    seq = Sequence("MOT17-02", root=mot_root)
    save_sample(seq, tmp_path / "fig" / "s.jpg", frame=3)
    assert (tmp_path / "fig" / "s.jpg").stat().st_size > 0
    img = seq.read_frame(1)
    rows = np.array([[1, 7, 10, 10, 20, 30, 1, -1, -1]], dtype=float)
    out = draw_tracks(img, rows)
    assert out.shape == img.shape and (out != img).any() and (img == 40).all()
    assert id_color(7) == id_color(7)
