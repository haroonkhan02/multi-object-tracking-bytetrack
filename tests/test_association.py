import numpy as np
import pytest

from mot_track.association import fuse_score, iou_cost, linear_assignment


def test_iou_cost_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], float)
    cost = iou_cost(a, a)
    assert cost[0, 0] == pytest.approx(0.0)
    assert cost[0, 1] == pytest.approx(1.0)


def test_linear_assignment_finds_global_optimum_not_greedy():
    # Greedy would take (0,0)=0.1 and then be forced into (1,1)=0.9.
    cost = np.array([[0.1, 0.2], [0.3, 0.9]])
    matches, ur, uc = linear_assignment(cost, max_cost=1.0)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert ur == [] and uc == []


def test_linear_assignment_rejects_expensive_pairs():
    cost = np.array([[0.2, 0.95], [0.95, 0.9]])
    matches, ur, uc = linear_assignment(cost, max_cost=0.7)
    assert matches == [(0, 0)]
    assert ur == [1] and uc == [1]


def test_linear_assignment_empty():
    matches, ur, uc = linear_assignment(np.zeros((0, 3)), 0.5)
    assert matches == [] and ur == [] and uc == [0, 1, 2]


def test_fuse_score_penalises_low_confidence():
    cost = np.array([[0.2, 0.2]])
    fused = fuse_score(cost, np.array([0.9, 0.3]))
    assert fused[0, 0] < fused[0, 1]
    assert fused[0, 0] == pytest.approx(1 - 0.8 * 0.9)
