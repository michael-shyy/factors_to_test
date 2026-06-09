"""tests/test_lz.py — LZ76 因子族单测。"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/hysheng/project8')

from features.lz_complexity.lz76 import (
    lz76_normalized, rolling_lz76, lz76_subseries, _lz76_count
)
from features.lz_complexity.encoder import encode_binary, encode_six_fast, size_bucket
from features.lz_complexity.transition_matrix import (
    build_transition_matrix, kl_divergence_uniform, transition_kl_asym
)
from features.lz_complexity.dict_stability import dict_stability_daily
from features.lz_complexity.regime_classifier import rolling_vol_label, split_by_regime
from features.lz_complexity.granger import granger_direction
from features.lz_complexity.price_context import price_context_masks, lz_price_context
from features.lz_complexity.probe_lz import probe_lz_factors


# ── LZ76 核心 ────────────────────────────────────────────────────────────

def test_lz76_count_min():
    """重复序列的 LZ 计数应远小于随机序列。"""
    seq_rep  = np.tile(np.array([0], dtype=np.int8), 50)
    seq_rand = np.array([0,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,
                         1,0,0,1,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,1,
                         0,1,0,0,1,0,1,1,0,1], dtype=np.int8)
    assert _lz76_count(seq_rep) < _lz76_count(seq_rand)


def test_lz76_count_max():
    """随机序列复杂度高于重复序列。"""
    rng = np.random.default_rng(42)
    seq_rand = rng.integers(0, 2, 100).astype(np.int8)
    seq_rep  = np.tile(np.array([0, 1], dtype=np.int8), 50)
    assert _lz76_count(seq_rand) > _lz76_count(seq_rep)


def test_lz76_normalized_range():
    """归一化值应在 (0, ∞) 范围，实践中通常 < 2。"""
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 2, 200).astype(np.int8)
    val = lz76_normalized(seq)
    assert 0 < val < 5


def test_lz76_normalized_short_returns_nan():
    seq = np.array([1, 0, 1], dtype=np.int8)
    assert np.isnan(lz76_normalized(seq))


def test_rolling_lz76_no_lookahead():
    """rolling_lz76[i] 只使用 seq[i-window:i]，结果长度等于输入。"""
    rng = np.random.default_rng(7)
    seq = rng.integers(0, 2, 600).astype(np.int8)
    out = rolling_lz76(seq, window=200)
    assert len(out) == len(seq)
    assert np.all(np.isnan(out[:199]))   # 前 window-1 个 = nan
    assert np.isfinite(out[199])


def test_rolling_lz76_deterministic():
    """相同输入两次结果一致。"""
    rng = np.random.default_rng(3)
    seq = rng.integers(0, 2, 300).astype(np.int8)
    a = rolling_lz76(seq, window=100)
    b = rolling_lz76(seq, window=100)
    np.testing.assert_array_equal(a, b)


# ── 编码 ────────────────────────────────────────────────────────────────

def test_encode_binary():
    bs = np.array([0, 1, 0, 0, 1], dtype=np.float64)
    enc = encode_binary(bs)
    np.testing.assert_array_equal(enc, [1, 0, 1, 1, 0])


def test_encode_six_range():
    rng = np.random.default_rng(1)
    bs = rng.integers(0, 2, 100).astype(np.float64)
    vol = rng.uniform(100, 10000, 100)
    enc = encode_six_fast(bs, vol)
    assert enc.min() >= 0
    assert enc.max() <= 5


def test_size_bucket_distribution():
    vol = np.arange(100, dtype=float)
    b = size_bucket(vol)
    assert (b == 0).sum() > 0
    assert (b == 2).sum() > 0
    assert set(b).issubset({0, 1, 2})


# ── 转移矩阵 ─────────────────────────────────────────────────────────────

def test_transition_matrix_rows_sum_to_one():
    rng = np.random.default_rng(2)
    seq = rng.integers(0, 6, 300).astype(np.int8)
    tm = build_transition_matrix(seq, k=6)
    np.testing.assert_allclose(tm.sum(axis=1), np.ones(6), atol=1e-6)


def test_kl_uniform_is_zero_for_uniform():
    """均匀矩阵对均匀分布的 KL 应为 0。"""
    k = 6
    tm = np.full((k, k), 1.0 / k)
    kl = kl_divergence_uniform(tm)
    assert abs(kl) < 1e-6


def test_transition_kl_asym_shapes():
    rng = np.random.default_rng(5)
    seq = rng.integers(0, 6, 400).astype(np.int8)
    kl_t, kl_b, kl_s = transition_kl_asym(seq)
    # 全部应为有限正数
    for v in (kl_t, kl_b, kl_s):
        assert np.isfinite(v) and v >= 0


# ── 词典稳定性 ───────────────────────────────────────────────────────────

def test_dict_stability_output():
    rng = np.random.default_rng(9)
    seq = rng.integers(0, 6, 1000).astype(np.int8)
    # 模拟 4 小时数据（每小时 250 笔）
    # 时间戳从 09:30 开始
    t0 = 9 * 3600 + 30 * 60
    ts = np.linspace(t0, t0 + 4 * 3600, 1000) + 1735780000  # UNIX
    mean_j, std_j = dict_stability_daily(seq, ts)
    assert 0.0 <= mean_j <= 1.0
    assert std_j >= 0.0


def test_dict_stability_short_returns_nan():
    seq = np.array([0, 1, 0, 1], dtype=np.int8)
    ts  = np.array([1e9, 1e9 + 1, 1e9 + 2, 1e9 + 3])
    m, s = dict_stability_daily(seq, ts)
    assert np.isnan(m) and np.isnan(s)


# ── Regime 分类 ──────────────────────────────────────────────────────────

def test_rolling_vol_label_values():
    rng = np.random.default_rng(11)
    prices = np.cumsum(rng.normal(0, 0.01, 500)) + 10.0
    ts = np.arange(500) * 4.0 + 1735780000
    labels = rolling_vol_label(prices, ts)
    assert set(np.unique(labels)).issubset({-1, 0, 1})


def test_split_by_regime():
    seq = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    labels = np.array([1, 0, 1, 0, 1, -1, 0, 1], dtype=np.int8)
    hi, lo = split_by_regime(seq, labels)
    assert len(hi) == (labels == 1).sum()
    assert len(lo) == (labels == 0).sum()


# ── Granger ──────────────────────────────────────────────────────────────

def test_granger_direction_returns_valid():
    rng = np.random.default_rng(13)
    lz_buy  = rng.uniform(0.5, 1.0, 60)
    lz_sell = rng.uniform(0.5, 1.0, 60)
    result = granger_direction(lz_buy, lz_sell)
    assert result in (-1, 0, 1)


def test_granger_short_returns_zero():
    lz_b = np.array([0.5, 0.6, 0.7])
    lz_s = np.array([0.5, 0.6, 0.7])
    assert granger_direction(lz_b, lz_s) == 0


# ── 探价前置 LZ ──────────────────────────────────────────────────────────

def test_probe_lz_keys():
    rng = np.random.default_rng(17)
    n = 300
    seq   = rng.integers(0, 2, n).astype(np.int8)
    bs    = rng.integers(0, 2, n).astype(np.float64)
    vol   = rng.uniform(100, 5000, n)
    probe = rng.integers(0, 2, n).astype(np.int8)
    res = probe_lz_factors(seq, bs, vol, probe, window=100)
    assert 'LZ_probe' in res
    assert 'LZ_burst_type' in res
    assert 'LZ_probe_diff' in res


# ── 价格背景条件化 LZ ────────────────────────────────────────────────────

def test_price_context_masks_lengths():
    rng = np.random.default_rng(19)
    prices = np.cumsum(rng.normal(0, 0.01, 500)) + 10.0
    vols   = rng.uniform(100, 1000, 500)
    masks = price_context_masks(prices, vols)
    for key, m in masks.items():
        assert len(m) == 500, f"{key} mask length mismatch"


def test_lz_price_context_returns_dict():
    rng = np.random.default_rng(21)
    seq  = rng.integers(0, 2, 500).astype(np.int8)
    prices = np.cumsum(rng.normal(0, 0.01, 500)) + 10.0
    vols   = rng.uniform(100, 1000, 500)
    masks = price_context_masks(prices, vols)
    result = lz_price_context(seq, masks, window=100)
    for k in ('LZ_upper', 'LZ_lower', 'LZ_price_context_diff',
              'LZ_vwap_regime', 'LZ_new_high_regime'):
        assert k in result
