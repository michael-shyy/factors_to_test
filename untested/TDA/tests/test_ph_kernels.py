"""test_ph_kernels.py：ripser 调用 + PD 后处理验证。"""
import numpy as np
import pytest
from ph_factors.ph_kernels import compute_ph
from ph_factors.landmark import maxmin_landmark


def _circle_cloud(n=200, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    pts += rng.normal(0, noise, pts.shape)
    return pts.astype(np.float64)


def test_h1_detects_circle():
    """noisy circle 应有一个显著 H1 环。"""
    cloud = _circle_cloud(n=300)
    _, _, dist = maxmin_landmark(cloud, M=80, seed=0)
    pd_h0, pd_h1 = compute_ph(dist)
    assert len(pd_h1) >= 1
    maxpers = (pd_h1[:, 1] - pd_h1[:, 0]).max()
    assert maxpers > 0.5, f"H1 MAXPERS too small: {maxpers}"


def test_no_inf_in_h1():
    cloud = _circle_cloud()
    _, _, dist = maxmin_landmark(cloud, M=60, seed=0)
    _, pd_h1 = compute_ph(dist)
    assert np.isfinite(pd_h1).all()


def test_h0_no_global_component():
    """H0 不应包含 (0, inf) 全局连通团。"""
    cloud = _circle_cloud()
    _, _, dist = maxmin_landmark(cloud, M=60, seed=0)
    pd_h0, _ = compute_ph(dist)
    assert np.isfinite(pd_h0).all()


def test_empty_dist():
    dist = np.empty((0, 0), dtype=np.float64)
    pd_h0, pd_h1 = compute_ph(dist)
    assert pd_h0.shape[0] == 0
    assert pd_h1.shape[0] == 0
