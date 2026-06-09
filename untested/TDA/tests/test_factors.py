"""test_factors.py：13 个 PD 统计量数值正确性验证。"""
import numpy as np
import pytest
from ph_factors.pd_statistics import compute_pd_stats

# 已知 PD：3 个点，持久性 [1.0, 2.0, 3.0]
_PD = np.array([[0.0, 1.0], [0.5, 2.5], [1.0, 4.0]])  # pers = [1, 2, 3]
_THETA = 0.5


def test_maxpers():
    s = compute_pd_stats(_PD, _THETA)
    assert s["MAXPERS"] == pytest.approx(3.0)


def test_top3_mean():
    s = compute_pd_stats(_PD, _THETA)
    assert s["TOP3_MEAN"] == pytest.approx(2.0)


def test_l2():
    s = compute_pd_stats(_PD, _THETA)
    assert s["L2"] == pytest.approx(1 + 4 + 9)


def test_maxpers_ratio():
    s = compute_pd_stats(_PD, _THETA)
    assert s["MAXPERS_RATIO"] == pytest.approx(3.0 / 6.0)


def test_count_low_high():
    s = compute_pd_stats(_PD, _THETA)
    # theta=0.5: COUNT_LOW = pers > 0.5 → all 3; COUNT_HIGH = pers > 1.5 → [2,3] = 2
    assert s["COUNT_LOW"] == 3.0
    assert s["COUNT_HIGH"] == 2.0


def test_empty_pd_all_nan():
    s = compute_pd_stats(np.empty((0, 2)), 1.0)
    for v in s.values():
        assert np.isnan(v)


def test_single_point_nan_fields():
    pd1 = np.array([[0.0, 1.0]])
    s = compute_pd_stats(pd1, 0.5)
    assert np.isnan(s["SKEW"])
    assert np.isnan(s["MAXGAP"])
    assert np.isnan(s["P75_MINUS_P25"])
    assert not np.isnan(s["MAXPERS"])


def test_zero_sum_pers():
    pd_zero = np.array([[1.0, 1.0], [2.0, 2.0]])  # pers = [0, 0]
    s = compute_pd_stats(pd_zero, 0.5)
    assert np.isnan(s["MAXPERS_RATIO"])
    assert s["ENT"] == pytest.approx(0.0)
