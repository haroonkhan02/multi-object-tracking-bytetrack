import numpy as np

from mot_track.mot_format import by_frame, load_mot_txt, read_seqinfo, split_gt, write_results


def test_read_seqinfo(mot_root):
    info = read_seqinfo(mot_root / "train" / "MOT17-02-FRCNN" / "seqinfo.ini")
    assert (info.length, info.width, info.height, info.fps) == (10, 200, 100, 30.0)


def test_load_missing_or_empty(tmp_path):
    assert load_mot_txt(tmp_path / "nope.txt").shape == (0, 9)
    (tmp_path / "empty.txt").write_text("")
    assert load_mot_txt(tmp_path / "empty.txt").shape == (0, 9)


def test_single_line_and_short_rows(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("1,2,3,4,5,6,0.5\n")
    rows = load_mot_txt(p)
    assert rows.shape == (1, 9)
    assert rows[0, 8] == -1  # missing visibility column is filled with -1


def test_split_gt_drops_conf0_and_separates_distractors(mot_root):
    gt = load_mot_txt(mot_root / "train" / "MOT17-02-FRCNN" / "gt" / "gt.txt")
    pedestrians, distractors = split_gt(gt)
    assert set(pedestrians[:, 1]) == {1, 2}
    assert set(distractors[:, 1]) == {3}
    assert 4 not in set(pedestrians[:, 1]) | set(distractors[:, 1])


def test_write_then_read_sorted(tmp_path):
    rows = np.array([[2, 1, 1, 2, 3, 4, 0.9], [1, 5, 1, 2, 3, 4, 0.8], [1, 2, 1, 2, 3, 4, 0.7]])
    write_results(tmp_path / "out" / "r.txt", rows)
    back = load_mot_txt(tmp_path / "out" / "r.txt")
    assert back[:, 0].tolist() == [1, 1, 2]
    assert back[:, 1].tolist() == [2, 5, 1]
    assert back[0, 7] == -1


def test_by_frame():
    rows = np.array([[3, 1, 0, 0, 1, 1, 1], [1, 2, 0, 0, 1, 1, 1], [3, 3, 0, 0, 1, 1, 1]])
    groups = by_frame(rows)
    assert sorted(groups) == [1, 3]
    assert len(groups[3]) == 2
    assert by_frame(np.zeros((0, 7))) == {}
