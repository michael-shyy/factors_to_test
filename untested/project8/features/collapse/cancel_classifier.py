"""cancel_classifier.py — 撤单细粒度因子。

从 lifecycle（build_order_lifecycle 输出）计算：
全撤/部撤比例、大单撤单速度、短单撤单比例、撤单-成交竞争速度。

时间戳规则：lifecycle 字段均为历史已发生事件，cancel_time ≤ 当前时刻，无前视。
"""
import numpy as np
import pandas as pd


def cancel_factors(
    lifecycle: pd.DataFrame,
    tick: pd.DataFrame,
    window_sec: float = 60.0,
) -> dict[str, float]:
    """
    Parameters
    ----------
    lifecycle : build_order_lifecycle 输出，需含：
                sysid, BSFlag, OrderTime, OrderVolume, OrderPrice,
                cancel_time, order_type_tag, filled_ratio, is_large_order,
                is_short_order（可选）
    tick      : tick 快照（含 time/bdv1/akv1）
    window_sec: 撤单-成交竞争统计时间窗口

    Returns
    -------
    dict 含 CANCEL_RATIO_bid/ask/ASYM,
            FULL_CANCEL_RATIO_bid/ask,
            PART_CANCEL_RATIO_bid/ask,
            LARGE_CANCEL_SPEED_bid/ask,
            SHORT_CANCEL_RATIO_bid/ask,
            COMPETE_RATIO_bid/ask/ASYM
    """
    lc = lifecycle.copy()
    has_cancel = lc['cancel_time'].notna()
    cancelled  = lc[has_cancel]

    def _side(df, bs): return df[df['BSFlag'] == bs]
    def _ratio(num, den): return float(num / den) if den > 0 else np.nan

    bid_all = _side(lc, 0)
    ask_all = _side(lc, 1)
    bid_can = _side(cancelled, 0)
    ask_can = _side(cancelled, 1)

    # 总消耗量（成交 + 撤单）
    bid_total_vol = (bid_all['OrderVolume'].sum() + 1e-8)
    ask_total_vol = (ask_all['OrderVolume'].sum() + 1e-8)

    # 撤单量占比
    cancel_ratio_bid = _ratio(bid_can['OrderVolume'].sum(), bid_total_vol)
    cancel_ratio_ask = _ratio(ask_can['OrderVolume'].sum(), ask_total_vol)

    # 全撤 vs 部撤
    def _full_cancel_mask(df):
        return df['order_type_tag'] == 'full_cancel'
    def _part_cancel_mask(df):
        return df['order_type_tag'] == 'partial_cancel'

    full_bid = bid_can[_full_cancel_mask(bid_can)]
    full_ask = ask_can[_full_cancel_mask(ask_can)]
    part_bid = bid_can[_part_cancel_mask(bid_can)]
    part_ask = ask_can[_part_cancel_mask(ask_can)]

    full_ratio_bid = _ratio(len(full_bid), max(len(bid_can), 1))
    full_ratio_ask = _ratio(len(full_ask), max(len(ask_can), 1))
    part_ratio_bid = _ratio(len(part_bid), max(len(bid_can), 1))
    part_ratio_ask = _ratio(len(part_ask), max(len(ask_can), 1))

    # 大单撤单速度（挂入到撤销的平均时长）
    def _large_cancel_speed(df):
        if 'is_large_order' not in df.columns:
            return np.nan
        lg = df[df['is_large_order'] == True]
        if len(lg) == 0:
            return np.nan
        speeds = (lg['cancel_time'] - lg['OrderTime']).values
        return float(np.nanmean(speeds[speeds > 0]))

    large_cancel_speed_bid = _large_cancel_speed(bid_can)
    large_cancel_speed_ask = _large_cancel_speed(ask_can)

    # 短单撤单比例
    def _short_cancel_ratio(df, all_df):
        if 'is_short_order' not in df.columns:
            return np.nan
        sh_can = df[df['is_short_order'] == True]
        sh_all = all_df[all_df.get('is_short_order', pd.Series(False, index=all_df.index)) == True] if 'is_short_order' in all_df.columns else pd.DataFrame()
        return _ratio(len(sh_can), max(len(sh_all), 1))

    short_cancel_bid = _short_cancel_ratio(bid_can, bid_all)
    short_cancel_ask = _short_cancel_ratio(ask_can, ask_all)

    # 撤单-成交竞争速度（tick 级别）
    compete = _compete_ratio(lc, tick, window_sec)

    def _d(a, b):
        return float(a - b) if not (np.isnan(a) or np.isnan(b)) else np.nan

    return {
        'CANCEL_RATIO_bid':         cancel_ratio_bid,
        'CANCEL_RATIO_ask':         cancel_ratio_ask,
        'CANCEL_RATIO_ASYM':        _d(cancel_ratio_bid, cancel_ratio_ask),
        'FULL_CANCEL_RATIO_bid':    full_ratio_bid,
        'FULL_CANCEL_RATIO_ask':    full_ratio_ask,
        'PART_CANCEL_RATIO_bid':    part_ratio_bid,
        'PART_CANCEL_RATIO_ask':    part_ratio_ask,
        'LARGE_CANCEL_SPEED_bid':   large_cancel_speed_bid,
        'LARGE_CANCEL_SPEED_ask':   large_cancel_speed_ask,
        'SHORT_CANCEL_RATIO_bid':   short_cancel_bid,
        'SHORT_CANCEL_RATIO_ask':   short_cancel_ask,
        'COMPETE_RATIO_bid':        compete['bid'],
        'COMPETE_RATIO_ask':        compete['ask'],
        'COMPETE_RATIO_ASYM':       _d(compete['bid'], compete['ask']),
    }


def _compete_ratio(lc: pd.DataFrame, tick: pd.DataFrame, window_sec: float) -> dict:
    """
    对同一价格档位，统计 window_sec 内成交量 / 撤单量。
    简化实现：对全日聚合，买/卖分别计算。
    """
    filled = lc[lc['order_type_tag'] == 'filled']
    cancelled = lc[lc['cancel_time'].notna()]

    def _r(bs):
        f_vol = filled[filled['BSFlag'] == bs]['filled_vol'].sum()
        c_vol = cancelled[cancelled['BSFlag'] == bs]['OrderVolume'].sum()
        return float(f_vol / (c_vol + 1e-8))

    return {'bid': _r(0), 'ask': _r(1)}
