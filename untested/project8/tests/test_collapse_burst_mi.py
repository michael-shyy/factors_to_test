"""tests/test_collapse_burst_mi.py — Collapse / Burst-Silence / MI 单测。"""
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/hysheng/project8')
sys.path.insert(0, '/home/hysheng')

from features.collapse.event_detector import detect_collapse_events
from features.collapse.refill_tracker import compute_refill_speed
from features.collapse.cancel_classifier import cancel_factors
from features.collapse.cluster_stats import collapse_cv
from features.burst_silence.segmenter import segment_trades, split_by_direction
from features.burst_silence.size_sequence import classify_burst, burst_type_ratios, Segment
from features.burst_silence.silence_activity import silence_underwater
from features.mutual_info.ksg_estimator import ksg_mi, rolling_ksg_mi
from features.mutual_info.transfer_entropy import transfer_entropy


# ── fixtures ────────────────────────────────────────────────────────────

def _make_tick(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t0 = 1735783800
    times = t0 + np.arange(n) * 4.0
    prices = 10.0 + np.cumsum(rng.normal(0, 0.01, n))
    data = {'time': times, 'akp1': prices + 0.01, 'bdp1': prices - 0.01}
    for i in range(1, 11):
        data[f'bdv{i}'] = rng.uniform(100, 1000, n)
        data[f'akv{i}'] = rng.uniform(100, 1000, n)
    return pd.DataFrame(data)


def _make_trade(n=500, seed=1):
    rng = np.random.default_rng(seed)
    t0 = 1735783800
    return pd.DataFrame({
        'TradeTime':   t0 + np.sort(rng.uniform(0, 14400, n)),
        'TradeCode':   np.zeros(n, dtype=int),
        'BSFlag':      rng.integers(0, 2, n).astype(float),
        'TradePrice':  10.0 + rng.normal(0, 0.05, n),
        'TradeVolume': rng.choice([100, 500, 1000, 5000, 50000], n).astype(float),
        'BuyOrderID':  rng.integers(1000, 9999, n).astype(float),
        'SellOrderID': rng.integers(1000, 9999, n).astype(float),
    })


def _make_order(n=300, seed=2):
    rng = np.random.default_rng(seed)
    t0 = 1735783800
    return pd.DataFrame({
        'OrderTime':   t0 + np.sort(rng.uniform(0, 14400, n)),
        'sysid':       np.arange(n, dtype=float),
        'OrderPrice':  10.0 + rng.normal(0, 0.05, n),
        'OrderVolume': rng.choice([100, 500, 1000, 5000], n).astype(float),
        'BSFlag':      rng.integers(0, 2, n).astype(float),
        'OrderType':   np.full(n, 4.0),
    })


def _make_lifecycle(n=300, seed=3):
    rng = np.random.default_rng(seed)
    t0 = 1735783800
    tags = rng.choice(['filled', 'full_cancel', 'partial_cancel'], n)
    cancel_time = np.where(tags != 'filled',
                           t0 + rng.uniform(0, 14400, n), np.nan)
    return pd.DataFrame({
        'sysid':         np.arange(n, dtype=float),
        'OrderTime':     t0 + rng.uniform(0, 14400, n),
        'OrderPrice':    10.0 + rng.normal(0, 0.05, n),
        'OrderVolume':   rng.choice([100, 500, 1000, 5000], n).astype(float),
        'BSFlag':        rng.integers(0, 2, n).astype(float),
        'cancel_time':   cancel_time,
        'order_type_tag': tags,
        'filled_ratio':  rng.uniform(0, 1, n),
        'filled_vol':    rng.uniform(0, 5000, n),
        'is_large_order': rng.choice([True, False], n),
        'is_short_order': rng.choice([True, False], n),
    })


# ── Collapse ─────────────────────────────────────────────────────────────

def test_detect_collapse_events_columns():
    tick  = _make_tick()
    trade = _make_trade()
    order = _make_order()
    events = detect_collapse_events(tick, trade, order)
    for col in ('time', 'side', 'total_drop', 'trade_vol', 'cancel_vol'):
        assert col in events.columns


def test_detect_collapse_events_sides():
    tick  = _make_tick()
    trade = _make_trade()
    order = _make_order()
    events = detect_collapse_events(tick, trade, order)
    if len(events) > 0:
        assert set(events['side'].unique()).issubset({'bid', 'ask'})


def test_refill_speed_keys():
    tick  = _make_tick()
    trade = _make_trade()
    order = _make_order()
    events = detect_collapse_events(tick, trade, order)
    res = compute_refill_speed(events, tick)
    for k in ('resupply_bid', 'resupply_ask', 'COLLAPSE_ASYM'):
        assert k in res


def test_cancel_factors_keys():
    lc   = _make_lifecycle()
    tick = _make_tick()
    res = cancel_factors(lc, tick)
    for k in ('CANCEL_RATIO_bid', 'CANCEL_RATIO_ask', 'CANCEL_RATIO_ASYM',
              'FULL_CANCEL_RATIO_bid', 'COMPETE_RATIO_ASYM'):
        assert k in res


def test_collapse_cv_asym():
    events = pd.DataFrame({
        'time': [1e9, 1e9+5, 1e9+10, 1e9+2, 1e9+6, 1e9+12],
        'side': ['bid','bid','bid','ask','ask','ask'],
    })
    res = collapse_cv(events)
    assert 'COLLAPSE_CV_ASYM' in res
    assert np.isfinite(res['CV_bid'])
    assert np.isfinite(res['CV_ask'])


# ── Burst-Silence ─────────────────────────────────────────────────────────

def test_segmenter_all_segments_cover_all_trades():
    rng = np.random.default_rng(5)
    ts  = np.cumsum(rng.exponential(0.5, 200)) + 1e9
    vol = rng.uniform(100, 1000, 200)
    bs  = rng.integers(0, 2, 200).astype(int)
    segs = segment_trades(ts, vol, bs)
    assert len(segs) > 0
    total_n = sum(s.n for s in segs)
    assert total_n == len(ts)


def test_segmenter_alternates_kinds():
    rng = np.random.default_rng(7)
    ts  = np.cumsum(rng.exponential(0.5, 100)) + 1e9
    vol = np.ones(100)
    bs  = np.zeros(100, dtype=int)
    segs = segment_trades(ts, vol, bs)
    kinds = [s.kind for s in segs]
    for i in range(1, len(kinds)):
        assert kinds[i] != kinds[i-1], "相邻段类型不应相同"


def test_classify_burst_types():
    vols = np.array([100, 500, 5000, 10000, 50000], dtype=float)
    q_large = np.quantile(vols, 0.8)
    q_small = np.quantile(vols, 0.2)

    # 递进型
    seg_prog = Segment('burst', 0, 1, 5,
                        vol_seq=list(vols), bs_seq=[0]*5)
    assert classify_burst(seg_prog, q_large, q_small) == 'prog'

    # 爆破型（起始大单）
    seg_shock = Segment('burst', 0, 1, 3,
                         vol_seq=[50000, 100, 200], bs_seq=[0]*3)
    assert classify_burst(seg_shock, q_large, q_small) == 'shock'


def test_burst_type_ratios_sum_leq_one():
    rng = np.random.default_rng(9)
    ts  = np.cumsum(rng.exponential(0.1, 500)) + 1e9
    vol = rng.uniform(100, 50000, 500)
    bs  = rng.integers(0, 2, 500).astype(int)
    segs   = segment_trades(ts, vol, bs)
    bursts = [s for s in segs if s.kind == 'burst']
    res = burst_type_ratios(bursts, vol)
    total = res['prog_ratio'] + res['shock_ratio'] + res['decay_ratio']
    assert total <= 1.0 + 1e-9


def test_silence_activity_keys():
    rng  = np.random.default_rng(11)
    ts   = np.cumsum(rng.exponential(0.5, 300)) + 1e9
    vol  = rng.uniform(100, 5000, 300)
    bs   = rng.integers(0, 2, 300).astype(int)
    segs = segment_trades(ts, vol, bs)
    sils = [s for s in segs if s.kind == 'silence']
    order = _make_order()
    trade = _make_trade()
    res = silence_underwater(sils, order, trade)
    for k in ('SILENCE_ORDER_ACTIVITY', 'SILENCE_CANCEL_RATIO', 'SILENCE_FULLCANCEL_RATE'):
        assert k in res


# ── MI ───────────────────────────────────────────────────────────────────

def test_ksg_mi_nonnegative():
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, 100)
    y = x + rng.normal(0, 0.5, 100)
    assert ksg_mi(x, y) >= 0


def test_ksg_mi_correlated_gt_independent():
    rng = np.random.default_rng(15)
    x  = rng.normal(0, 1, 200)
    y_dep = x + rng.normal(0, 0.1, 200)
    y_ind = rng.normal(0, 1, 200)
    mi_dep = ksg_mi(x, y_dep)
    mi_ind = ksg_mi(x, y_ind)
    assert mi_dep > mi_ind


def test_ksg_mi_short_returns_nan():
    assert np.isnan(ksg_mi(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def test_rolling_ksg_mi_length_and_nan():
    rng = np.random.default_rng(17)
    x = rng.normal(0, 1, 120)
    y = x + rng.normal(0, 0.5, 120)
    out = rolling_ksg_mi(x, y, window=50, k=3)
    assert len(out) == len(x)
    assert np.all(np.isnan(out[:49]))
    assert np.isfinite(out[49])


def test_transfer_entropy_returns_three():
    rng = np.random.default_rng(19)
    x = rng.normal(0, 1, 100)
    y = np.roll(x, 1) + rng.normal(0, 0.5, 100)
    te_x2y, te_y2x, te_dir = transfer_entropy(x, y)
    # 至少一个有限值
    assert any(np.isfinite(v) for v in [te_x2y, te_y2x, te_dir])


def test_transfer_entropy_dir_sign():
    """X→Y 因果，TE_DIR 应倾向正值。"""
    rng = np.random.default_rng(21)
    x = rng.normal(0, 1, 300)
    y = np.empty(300)
    y[0] = 0
    for i in range(1, 300):
        y[i] = 0.8 * x[i-1] + rng.normal(0, 0.2)
    te_x2y, te_y2x, te_dir = transfer_entropy(x, y)
    if np.isfinite(te_dir):
        assert te_dir > 0
