"""Constant-velocity Kalman filter for bounding boxes.

State x = [cx, cy, a, h, vcx, vcy, va, vh]: box centre, aspect ratio (w / h), height and their
velocities. Measurements are z = [cx, cy, a, h]. This is the parameterisation DeepSORT and
ByteTrack use: process and measurement noise scale with the box height, so a 400 px tall person
near the camera is allowed to move more pixels per frame than a 60 px one far away.
"""

from __future__ import annotations

import numpy as np

NDIM = 4
DT = 1.0

# Noise as a fraction of the box height (values from the DeepSORT / ByteTrack reference code).
STD_POSITION = 1.0 / 20
STD_VELOCITY = 1.0 / 160

# 95 % quantile of the chi-square distribution with 4 degrees of freedom, used for gating.
CHI2_95_4DOF = 9.4877


def xywh_to_xyah(box: np.ndarray) -> np.ndarray:
    x, y, w, h = np.asarray(box, dtype=float)
    return np.array([x + w / 2, y + h / 2, w / max(h, 1e-6), h])


def xyah_to_xywh(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = np.asarray(xyah, dtype=float)[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, w, h])


class KalmanBoxFilter:
    """Linear Kalman filter shared by all tracks (it holds the model, tracks hold the state)."""

    def __init__(self) -> None:
        self.F = np.eye(2 * NDIM)
        self.F[:NDIM, NDIM:] = DT * np.eye(NDIM)
        self.H = np.eye(NDIM, 2 * NDIM)

    @staticmethod
    def _std(h: float, pos_scale: float, vel_scale: float) -> np.ndarray:
        pos = [pos_scale * h, pos_scale * h, 1e-2, pos_scale * h]
        vel = [vel_scale * h, vel_scale * h, 1e-5, vel_scale * h]
        return np.array(pos + vel)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """New track from one measurement: zero velocity, large velocity uncertainty."""
        mean = np.r_[z, np.zeros(NDIM)]
        std = self._std(z[3], 2 * STD_POSITION, 10 * STD_VELOCITY)
        return mean, np.diag(std**2)

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = np.diag(self._std(mean[3], STD_POSITION, STD_VELOCITY) ** 2)
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + Q
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """State distribution mapped to measurement space (adds measurement noise R)."""
        std = [STD_POSITION * mean[3], STD_POSITION * mean[3], 1e-1, STD_POSITION * mean[3]]
        R = np.diag(np.square(std))
        return self.H @ mean, self.H @ cov @ self.H.T + R

    def update(
        self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, S = self.project(mean, cov)
        # K = P H^T S^-1, solved instead of inverted for numerical stability.
        K = np.linalg.solve(S, (cov @ self.H.T).T).T
        innovation = np.asarray(z, dtype=float) - proj_mean
        mean = mean + K @ innovation
        cov = cov - K @ S @ K.T
        return mean, cov

    def gating_distance(self, mean: np.ndarray, cov: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance from the predicted box to each measurement."""
        proj_mean, S = self.project(mean, cov)
        d = np.asarray(zs, dtype=float).reshape(-1, NDIM) - proj_mean
        L = np.linalg.cholesky(S)
        y = np.linalg.solve(L, d.T)
        return np.sum(y**2, axis=0)
