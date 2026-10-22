import numpy as np
import pytest

from mot_track.kalman import CHI2_95_4DOF, KalmanBoxFilter, xyah_to_xywh, xywh_to_xyah


def test_xyah_round_trip():
    box = np.array([10.0, 20.0, 30.0, 60.0])
    assert xywh_to_xyah(box) == pytest.approx([25.0, 50.0, 0.5, 60.0])
    assert xyah_to_xywh(xywh_to_xyah(box)) == pytest.approx(box)


def test_filter_learns_constant_velocity():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(xywh_to_xyah([0, 0, 20, 40]))
    for k in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, xywh_to_xyah([3.0 * k, 1.0 * k, 20, 40]))
    assert mean[4] == pytest.approx(3.0, abs=0.05)
    assert mean[5] == pytest.approx(1.0, abs=0.05)
    pred, _ = kf.predict(mean, cov)
    assert pred[0] == pytest.approx(10 + 3.0 * 30, abs=0.3)


def test_covariance_shrinks_with_updates_and_grows_with_predictions():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(xywh_to_xyah([0, 0, 20, 40]))
    pred_mean, pred_cov = kf.predict(mean, cov)
    _, upd_cov = kf.update(pred_mean, pred_cov, xywh_to_xyah([0, 0, 20, 40]))
    assert np.trace(pred_cov) > np.trace(cov)
    assert np.trace(upd_cov) < np.trace(pred_cov)
    assert np.allclose(upd_cov, upd_cov.T)


def test_gating_separates_near_and_far_measurements():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(xywh_to_xyah([100, 100, 20, 40]))
    near = xywh_to_xyah([101, 100, 20, 40])
    far = xywh_to_xyah([300, 100, 20, 40])
    d = kf.gating_distance(mean, cov, np.stack([near, far]))
    assert d[0] < CHI2_95_4DOF < d[1]
