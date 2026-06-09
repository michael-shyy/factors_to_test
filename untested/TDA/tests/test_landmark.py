"""test_landmark.py：maxmin landmark 正确性验证。"""
import numpy as np
import pytest
from ph_factors.landmark import maxmin_landmark


def test_returns_m_points():
    cloud = np.random.default_rng(0).random((1000, 2))
    lm, idx, dist = maxmin_landmark(cloud, M=50, seed=0)
    assert lm.shape == (50, 2)
    assert idx.shape == (50,)
    assert dist.shape == (50, 50)


def test_dist_matrix_symmetric():
    cloud = np.random.default_rng(1).random((200, 3))
    _, _, dist = maxmin_landmark(cloud, M=30, seed=1)
    np.testing.assert_allclose(dist, dist.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-12)


def test_reproducible():
    cloud = np.random.default_rng(2).random((500, 2))
    _, idx1, _ = maxmin_landmark(cloud, M=40, seed=42)
    _, idx2, _ = maxmin_landmark(cloud, M=40, seed=42)
    np.testing.assert_array_equal(idx1, idx2)


def test_n_less_than_m():
    cloud = np.random.default_rng(3).random((10, 2))
    lm, idx, dist = maxmin_landmark(cloud, M=50, seed=0)
    assert lm.shape[0] == 10


def test_empty_cloud():
    cloud = np.empty((0, 2))
    lm, idx, dist = maxmin_landmark(cloud, M=10, seed=0)
    assert lm.shape[0] == 0
