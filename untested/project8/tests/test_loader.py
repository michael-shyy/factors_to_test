import pytest
import pandas as pd
import sys
sys.path.insert(0, '/home/hysheng/project8')

from data.loader import load_daily, get_file_path


def test_get_file_path():
    p = get_file_path('20250102', 'sh600000')
    assert '2025' in p and '20250102' in p and 'sh600000.h5' in p


def test_load_daily_schema():
    order, tick, trade = load_daily('20250102', 'sh600000')

    assert set(['OrderTime', 'sysid', 'OrderPrice', 'OrderVolume',
                'BSFlag', 'OrderType']).issubset(order.columns)
    assert 'time' in tick.columns
    assert all(f in tick.columns for f in ['bdp1', 'bdv1', 'akp1', 'akv1'])
    assert set(['TradeTime', 'TradeCode', 'BSFlag',
                'TradePrice', 'TradeVolume',
                'SellOrderID', 'BuyOrderID']).issubset(trade.columns)


def test_load_daily_dtypes():
    order, tick, trade = load_daily('20250102', 'sh600000')
    assert order['OrderTime'].dtype == float
    assert tick['time'].dtype == float
    assert trade['TradeTime'].dtype == float


def test_load_daily_nonempty():
    order, tick, trade = load_daily('20250102', 'sh600000')
    assert len(order) > 0
    assert len(tick) > 0
    assert len(trade) > 0
