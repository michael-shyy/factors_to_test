"""test_signals.py：主信号计算正确性验证。"""
import numpy as np
import pandas as pd
import pytest
from ph_factors.signals import (
    compute_ob_imbal, compute_dp_micro, compute_spread,
    compute_depth5, segment_mask,
)


def _make_snapshot(n=10):
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="3s")
    data = {
        **{f"bid_price_{i}": np.ones(n) * (100 - i * 0.01) for i in range(1, 6)},
        **{f"ask_price_{i}": np.ones(n) * (100 + i * 0.01) for i in range(1, 6)},
        **{f"bid_vol_{i}": np.ones(n) * 100 for i in range(1, 6)},
        **{f"ask_vol_{i}": np.ones(n) * 80 for i in range(1, 6)},
    }
    return pd.DataFrame(data, index=idx)


def test_ob_imbal_range():
    snap = _make_snapshot()
    imbal = compute_ob_imbal(snap)
    assert (imbal.between(-1, 1)).all()


def test_ob_imbal_positive_when_bid_gt_ask():
    snap = _make_snapshot()
    # bid_vol > ask_vol → imbal > 0
    assert (compute_ob_imbal(snap) > 0).all()


def test_dp_micro_first_nan():
    snap = _make_snapshot()
    dp = compute_dp_micro(snap)
    assert np.isnan(dp.iloc[0])


def test_spread_positive():
    snap = _make_snapshot()
    spread = compute_spread(snap)
    assert (spread > 0).all()


def test_depth5_sum():
    snap = _make_snapshot()
    depth = compute_depth5(snap)
    # 5 档 × (100 + 80) = 900
    assert (depth == 900).all()


def test_segment_mask_full():
    idx = pd.date_range("2024-01-02 09:00", periods=200, freq="1min")
    mask = segment_mask(idx, "FULL")
    # 09:00 应排除，09:30 应包含，11:30-13:00 应排除
    times = idx.time
    assert not mask[0]  # 09:00
    assert mask[list(idx.strftime("%H:%M")).index("09:30")]


def test_segment_mask_open():
    idx = pd.date_range("2024-01-02 09:30", periods=60, freq="1min")
    mask = segment_mask(idx, "OPEN")
    # 10:30 之后应为 False
    after_open = idx.time > pd.Timestamp("10:30").time()
    assert not mask[after_open].any()
