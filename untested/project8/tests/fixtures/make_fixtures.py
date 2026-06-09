"""tests/fixtures/make_fixtures.py — 从实际 h5 文件截取小样本，供单测使用。"""
import os
import sys
import pandas as pd

sys.path.insert(0, '/home/hysheng/project8')
from data.loader import load_daily

OUT = os.path.dirname(os.path.abspath(__file__))
DATE, STOCK = '20250102', 'sh600000'

def make():
    order, tick, trade = load_daily(DATE, STOCK)
    # 取 09:30–09:35 的数据（约前 300 条 tick）
    t0 = tick['time'].min()
    t1 = t0 + 5 * 60
    tick_s  = tick[tick['time'] <= t1].head(100).reset_index(drop=True)
    t_range = (tick_s['time'].min(), tick_s['time'].max())
    order_s = order[(order['OrderTime'] >= t_range[0]) &
                    (order['OrderTime'] <= t_range[1])].reset_index(drop=True)
    trade_s = trade[(trade['TradeTime'] >= t_range[0]) &
                    (trade['TradeTime'] <= t_range[1])].reset_index(drop=True)

    tick_s.to_parquet(os.path.join(OUT, 'tick.parquet'))
    order_s.to_parquet(os.path.join(OUT, 'order.parquet'))
    trade_s.to_parquet(os.path.join(OUT, 'trade.parquet'))
    print(f"tick={len(tick_s)}, order={len(order_s)}, trade={len(trade_s)}")

if __name__ == '__main__':
    make()
