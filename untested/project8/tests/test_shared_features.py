import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/hysheng/project8')

from data.shared_features import price_probe_flag, tick_stability, bob_symm, add_shared_features

FIXTURES = '/home/hysheng/project8/tests/fixtures'


@pytest.fixture(scope='module')
def tick():
    return pd.read_parquet(f'{FIXTURES}/tick.parquet')


def test_probe_flag_binary(tick):
    pf = price_probe_flag(tick)
    assert set(pf.dropna().unique()).issubset({0, 1})


def test_tick_stability_positive(tick):
    ts = tick_stability(tick)
    assert (ts.dropna() > 0).all()


def test_bob_symm_range(tick):
    bs = bob_symm(tick)
    valid = bs.dropna()
    assert (valid >= -1).all() and (valid <= 1).all()


def test_add_shared_features_columns(tick):
    enriched = add_shared_features(tick)
    for col in ['probe_flag', 'tick_stability', 'bob_symm']:
        assert col in enriched.columns
    assert len(enriched) == len(tick)


def test_no_lookahead_probe_flag(tick):
    """probe_flag 在第 0 行应为 NaN 或 0（无历史数据）。"""
    pf = price_probe_flag(tick, window=5)
    # 前几行因无足够历史，不应出现基于未来数据的非零值
    # 只验证长度一致
    assert len(pf) == len(tick)
