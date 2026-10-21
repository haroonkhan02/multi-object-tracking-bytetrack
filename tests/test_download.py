import zipfile

from mot_track.download import extract_train, wanted


def test_wanted_selects_training_gt_det_and_one_set_of_images():
    base = "MOT17/train/MOT17-02-"
    assert wanted(base + "FRCNN/img1/000001.jpg")
    assert wanted(base + "FRCNN/gt/gt.txt")
    assert wanted(base + "DPM/det/det.txt")
    assert wanted(base + "SDP/seqinfo.ini")
    assert not wanted(base + "DPM/img1/000001.jpg")  # duplicate images
    assert not wanted(base + "DPM/gt/gt.txt")
    assert not wanted("MOT17/test/MOT17-01-FRCNN/img1/000001.jpg")
    assert not wanted("MOT17/train/MOT17-01-FRCNN/img1/000001.jpg")  # not a train sequence
    assert not wanted("MOT17/train/MOT17-02-FRCNN/")


def test_extract_train(tmp_path):
    z = tmp_path / "MOT17.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("MOT17/train/MOT17-02-FRCNN/img1/000001.jpg", b"abc")
        zf.writestr("MOT17/train/MOT17-02-FRCNN/gt/gt.txt", "1,1,0,0,1,1,1,1,1\n")
        zf.writestr("MOT17/train/MOT17-02-DPM/img1/000001.jpg", b"dup")
        zf.writestr("MOT17/test/MOT17-01-FRCNN/img1/000001.jpg", b"test")
    out = tmp_path / "out"
    assert extract_train(z, out) == 2
    assert (out / "train" / "MOT17-02-FRCNN" / "img1" / "000001.jpg").read_bytes() == b"abc"
    assert not (out / "test").exists()
    assert extract_train(z, out) == 0  # re-running skips files that are already there
