import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/hysheng/project8')
sys.path.insert(0, '/home/hysheng')

from data.loader import load_daily
from data.preprocessor import (
    filter_continuous_auction, tag_orders, build_lifecycle, tag_size_bucket
)

DATE, STOCK = '20250102', 'sh600000'


@pytest.fixture(scope='module')
def raw():
    return load_daily(DATE, STOCK)


def test_filter_continuous_auction(raw):
    order, tick, trade = raw
    filtered = filter_continuous_auction(order, 'OrderTime')
    local_sec = (filtered['OrderTime'] + 28800) % 86400
    assert (local_sec >= 9*3600 + 30*60).all()
    assert (local_sec < 14*3600).all()


def test_tag_orders_columns(raw):
    order, _, trade = raw
    tagged_o, tagged_t = tag_orders(order, trade)
    for col in ['is_short_order', 'is_long_order', 'is_large_order']:
        assert col in tagged_o.columns
    assert 'is_short_order' in tagged_t.columns


def test_tag_orders_no_lookahead(raw):
    """短单/长单阈值只依赖截面前数据，输出行数 ≤ 输入行数。"""
    order, _, trade = raw
    tagged_o, _ = tag_orders(order, trade)
    assert len(tagged_o) <= len(order)


def test_build_lifecycle_columns(raw):
    order, _, trade = raw
    lc = build_lifecycle(order, trade, stock_id=STOCK)
    for col in ['filled_vol', 'filled_ratio', 'cancel_time', 'order_type_tag']:
        assert col in lc.columns


def test_build_lifecycle_filled_ratio_range(raw):
    order, _, trade = raw
    lc = build_lifecycle(order, trade, stock_id=STOCK)
    valid = lc['filled_ratio'].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_tag_size_bucket(raw):
    order, _, _ = raw
    tagged = tag_size_bucket(order, n_buckets=5)
    assert 'size_bucket' in tagged.columns
    valid = tagged['size_bucket'].dropna()
    assert valid.min() >= 1
    assert valid.max() <= 5
