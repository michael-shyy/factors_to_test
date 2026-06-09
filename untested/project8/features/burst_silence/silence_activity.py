"""silence_activity.py — 静默期水下活动（委托/撤单活跃度，全向量化）。"""
import numpy as np
import pandas as pd
from .segmenter import Segment

_NAN = {'SILENCE_ORDER_ACTIVITY': np.nan, 'SILENCE_CANCEL_RATIO': np.nan, 'SILENCE_FULLCANCEL_RATE': np.nan}


def silence_underwater(
    silence_segs: list[Segment],
    order: pd.DataFrame,
    trade: pd.DataFrame,
) -> dict[str, float]:
    if not silence_segs:
        return _NAN

    starts = np.array([s.start for s in silence_segs])
    ends   = np.array([s.end   for s in silence_segs])
    durs   = np.maximum(ends - starts, 1e-3)

    order_times  = np.sort(order['OrderTime'].values)
    cancel_times = np.sort(
        trade.loc[trade['TradeCode'] == 4, 'TradeTime'].values
        if len(trade) > 0 else np.array([])
    )

    # 用 searchsorted 批量统计每段内事件数
    o_lo = np.searchsorted(order_times, starts, side='left')
    o_hi = np.searchsorted(order_times, ends,   side='right')
    n_orders = o_hi - o_lo

    if len(cancel_times):
        c_lo = np.searchsorted(cancel_times, starts, side='left')
        c_hi = np.searchsorted(cancel_times, ends,   side='right')
        n_cancels = c_hi - c_lo
    else:
        n_cancels = np.zeros(len(silence_segs), dtype=int)

    total_dur    = durs.sum()
    if total_dur == 0:
        return _NAN

    total_events = n_orders.sum() + n_cancels.sum()
    activity_rate = total_events / total_dur
    cancel_ratio  = n_cancels.sum() / total_events if total_events > 0 else np.nan
    # full_cancel == cancel (简化，同原逻辑)
    full_cancel_rate = float(n_cancels.sum() / n_cancels.sum()) if n_cancels.sum() > 0 else np.nan

    return {
        'SILENCE_ORDER_ACTIVITY':  float(activity_rate),
        'SILENCE_CANCEL_RATIO':    float(cancel_ratio),
        'SILENCE_FULLCANCEL_RATE': float(full_cancel_rate),
    }
