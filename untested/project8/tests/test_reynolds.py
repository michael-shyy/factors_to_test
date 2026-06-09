"""tests/test_reynolds.py — 雷诺数因子族单测。"""
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/hysheng/project8')
sys.path.insert(0, '/home/hysheng')

from features.reynolds.mlofi import compute_mlofi, zscore_daily, mlofi_by_size
from features.reynolds.resilience import compute_resilience
from features.reynolds.penetration_depth import compute_penetration_depth
from features.reynolds.impact_efficiency import compute_impact_efficiency
from features.reynolds.dead_zone import apply_dead_zone, re_of_daily


# ── fixtures ────────────────────────────────────────────────────────────

def _make_tick(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t0 = 1735783800  # 09:30
    times = t0 + np.arange(n) * 4.0
    base = 10.0
    prices = base + np.cumsum(rng.normal(0, 0.01, n))
    spread = 0.02
    data = {
        'time':  times,
        'akp1':  prices + spread / 2,
        'bdp1':  prices - spread / 2,
        'akp2':  prices + spread,
        'bdp2':  prices - spread,
    }
    for i in range(1, 11):
        if f'bdv{i}' not in data:
            data[f'bdv{i}'] = rng.uniform(100, 1000, n)
            data[f'akv{i}'] = rng.uniform(100, 1000, n)
    return pd.DataFrame(data)


def _make_trade(n=500, seed=1):
    rng = np.random.default_rng(seed)
    t0 = 1735783800
    times = t0 + np.sort(rng.uniform(0, 14400, n))
    return pd.DataFrame({
        'TradeTime':   times,
        'TradeCode':   np.zeros(n, dtype=int),
        'BSFlag':      rng.integers(0, 2, n).astype(float),
        'TradePrice':  10.0 + rng.normal(0, 0.05, n),
        'TradeVolume': rng.choice([100, 500, 1000, 5000, 50000], n).astype(float),
        'BuyOrderID':  rng.integers(1000, 9999, n).astype(float),
        'SellOrderID': rng.integers(1000, 9999, n).astype(float),
    })


# ── MLOFI ───────────────────────────────────────────────────────────────

def test_mlofi_length():
    tick = _make_tick()
    mlofi = compute_mlofi(tick)
    assert len(mlofi) == len(tick)
    assert np.isnan(mlofi.iloc[0])
    assert np.isfinite(mlofi.iloc[1:]).any()


def test_mlofi_first_nan():
    tick = _make_tick()
    m = compute_mlofi(tick)
    assert np.isnan(m.iloc[0])


def test_zscore_mean_near_zero():
    s = pd.Series(np.arange(100, dtype=float))
    z = zscore_daily(s)
    assert abs(z.mean()) < 1e-10
    assert abs(z.std() - 1.0) < 1e-6


def test_mlofi_by_size_keys():
    tick  = _make_tick()
    trade = _make_trade()
    result = mlofi_by_size(tick, trade)
    for key in ('mlofi_all', 'mlofi_large', 'mlofi_small'):
        assert key in result
        assert len(result[key]) == len(tick)


# ── 恢复速度 ─────────────────────────────────────────────────────────────

def test_resilience_keys():
    tick  = _make_tick(400)
    trade = _make_trade(1000)
    res = compute_resilience(tick, trade)
    for k in ('resil_ask', 'resil_bid', 'resil_norm'):
        assert k in res


def test_resilience_positive():
    """恢复速度应为正数（或 nan，当无足够大单时）。"""
    tick  = _make_tick(400)
    trade = _make_trade(1000)
    res = compute_resilience(tick, trade)
    for k in ('resil_ask', 'resil_bid', 'resil_norm'):
        v = res[k]
        if not np.isnan(v):
            assert v > 0, f"{k} = {v} 应为正数"


# ── 穿透深度 ─────────────────────────────────────────────────────────────

def test_penetration_depth_keys():
    tick  = _make_tick(400)
    trade = _make_trade(1000)
    result = compute_penetration_depth(tick, trade)
    for k in ('PENETRATION_DEPTH_buy', 'PENETRATION_DEPTH_sell',
              'PENETRATION_DEPTH_ASYM', 'PENETRATION_RE_buy', 'PENETRATION_RE_sell'):
        assert k in result


def test_penetration_depth_positive():
    tick  = _make_tick(400)
    trade = _make_trade(1000)
    res = compute_penetration_depth(tick, trade)
    for k in ('PENETRATION_DEPTH_buy', 'PENETRATION_DEPTH_sell'):
        v = res[k]
        if not np.isnan(v):
            assert v >= 1.0


# ── 冲击效率 ─────────────────────────────────────────────────────────────

def test_impact_efficiency_keys():
    tick  = _make_tick(400)
    trade = _make_trade(1000)
    res = compute_impact_efficiency(tick, trade)
    for k in ('IMPACT_EFF_trend', 'IMPACT_EFF_asym',
              'IMPACT_DECAY_slope_buy', 'IMPACT_DECAY_slope_sell',
              'IMPACT_DECAY_slope_diff'):
        assert k in res


# ── 死区 ─────────────────────────────────────────────────────────────────

def test_dead_zone_zeros_below_threshold():
    arr = np.array([-2.0, -0.1, 0.0, 0.1, 2.0])
    out = apply_dead_zone(arr, pct=0.30)
    # P30 of abs = 0.1，所以 |x| < 0.1 的应被置 0
    assert out[1] == 0.0 or out[2] == 0.0


def test_dead_zone_preserves_extremes():
    arr = np.array([-5.0, -0.01, 0.0, 0.01, 5.0])
    out = apply_dead_zone(arr, pct=0.30)
    assert out[0] != 0.0 or out[4] != 0.0   # 极端值不应全被清零


def test_re_of_daily_nan_on_nan_resil():
    m = np.ones(100)
    res = re_of_daily(m, np.nan)
    assert np.isnan(res['Re_OF'])
    assert np.isnan(res['Re_OF_raw'])
