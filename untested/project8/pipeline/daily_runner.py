"""pipeline/daily_runner.py — 单日单股因子计算入口。"""
import os
import sys
sys.path.insert(0, '/home/hysheng/project8')
sys.path.insert(0, '/home/hysheng')

import pandas as pd
import numpy as np
from data.loader import load_daily
from data.context import DailyDataContext
from features.lz_complexity.factor_builder import build_lz_factors
from features.reynolds.factor_builder import build_reynolds_factors
from features.collapse.factor_builder import build_collapse_factors
from features.burst_silence.factor_builder import build_burst_silence_factors
from features.mutual_info.factor_builder import build_mi_factors


_BUILDERS = [
    build_lz_factors,
    build_reynolds_factors,
    build_collapse_factors,
    build_burst_silence_factors,
    build_mi_factors,
]


def run_one(date: str, stock_id: str) -> dict | None:
    """
    计算单只股票单日所有因子。

    Returns
    -------
    dict 含 date, stock_id 和所有因子值，失败返回 None。
    """
    try:
        order, tick, trade = load_daily(date, stock_id)
        ctx = DailyDataContext(tick=tick, trade=trade, order=order, stock_id=stock_id)
        row: dict = {'date': date, 'stock_id': stock_id}
        for builder in _BUILDERS:
            row.update(builder(ctx))
        return row
    except Exception as e:
        print(f'[WARN] {date}/{stock_id}: {e}')
        return None
