import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/hysheng/project8')

from data.snapshot_aligner import align_orders_to_tick, snapshot_features

FIXTURES = '/home/hysheng/project8/tests/fixtures'


@pytest.fixture(scope='module')
def fixtures():
    tick  = pd.read_parquet(f'{FIXTURES}/tick.parquet')
    order = pd.read_parquet(f'{FIXTURES}/order.parquet')
    trade = pd.read_parquet(f'{FIXTURES}/trade.parquet')
    return order, tick, trade


def test_align_no_lookahead(fixtures):
    """对齐后，每笔委托的匹配 tick.time 必须 ≤ OrderTime。"""
    order, tick, trade = fixtures
    merged = align_orders_to_tick(order, tick)
    valid = merged.dropna(subset=['time'])
    assert (valid['time'] <= valid['OrderTime']).all(), \
        "存在前视：tick.time > OrderTime"


def test_snapshot_features_shape(fixtures):
    order, tick, trade = fixtures
    snap = snapshot_features(order, trade, tick)
    assert len(snap) == len(tick)
    for col in ['time', 'n_buy_order', 'n_sell_order',
                'buy_trade_vol', 'sell_trade_vol', 'running_vwap']:
        assert col in snap.columns


def test_running_vwap_monotone_denominator(fixtures):
    """running_vwap 的分母（累计成交量）必须单调递增。"""
    order, tick, trade = fixtures
    snap = snapshot_features(order, trade, tick)
    cum_vol = snap['buy_trade_vol'] + snap['sell_trade_vol']
    cum_vol_cumsum = cum_vol.cumsum()
    assert (cum_vol_cumsum.diff().dropna() >= 0).all()


def test_snapshot_window_no_overlap(fixtures):
    """每个快照窗口只包含 (t_prev, t] 的事件，不重复计入。"""
    order, tick, trade = fixtures
    snap = snapshot_features(order, trade, tick)
    # 所有窗口的 order 笔数之和应 ≤ 原始 order 总数
    total_orders = snap['n_buy_order'].sum() + snap['n_sell_order'].sum()
    assert total_orders <= len(order)
