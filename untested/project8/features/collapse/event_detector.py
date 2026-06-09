"""event_detector.py — 坍缩事件识别。

识别买/卖一档挂单量快速下降的"消耗事件"，拆分成交型 vs 撤单型消耗。

时间戳规则：
- 基于相邻两个快照差分，t 时刻消耗 = bdv1[t] - bdv1[t-1]
- 匹配成交/撤单记录时使用 (t_prev, t] 窗口，无前视
"""
import numpy as np
import pandas as pd


def detect_collapse_events(
    tick: pd.DataFrame,
    trade: pd.DataFrame,
    order: pd.DataFrame,
    threshold_pct: float = 0.5,
) -> pd.DataFrame:
    """
    扫描快照序列，识别买/卖一档挂单量快速下降事件，并拆分消耗来源。

    Parameters
    ----------
    tick           : tick 快照，含 time/bdv1/akv1
    trade          : 逐笔成交（含 TradeCode/TradeTime/TradeVolume/BSFlag）
    order          : 逐笔委托（含 OrderTime/OrderVolume/BSFlag/OrderType）
    threshold_pct  : 下降量 > 日均单笔成交量 P50 * threshold_pct 才触发事件

    Returns
    -------
    DataFrame，每行一个消耗事件，列：
      time, side(bid/ask), total_drop, trade_vol, cancel_vol,
      t_prev, large_cancel(bool), short_cancel(bool)
    """
    tick_s = tick.sort_values('time').reset_index(drop=True)
    real   = trade[trade['TradeCode'] == 0].sort_values('TradeTime').reset_index(drop=True)

    tick_times = tick_s['time'].values
    t_prevs = np.empty(len(tick_times))
    t_prevs[0] = tick_times[0] - 1
    t_prevs[1:] = tick_times[:-1]

    vol_median = np.nanmedian(real['TradeVolume'].values) if len(real) > 0 else 100.0
    threshold  = vol_median * threshold_pct

    real_times  = real['TradeTime'].values
    real_vol    = real['TradeVolume'].values
    real_bs     = real['BSFlag'].values

    # 上交所撤单（OrderType==5）或深交所撤单（TradeCode==4）
    cancel_trade = trade[trade['TradeCode'] == 4] if len(trade) > 0 else pd.DataFrame()
    cancel_order = order[order.get('OrderType', pd.Series(dtype=float)) == 5] if 'OrderType' in order.columns else pd.DataFrame()

    records = []
    for side, vol_col, bs_flag in [('bid', 'bdv1', 1), ('ask', 'akv1', 0)]:
        if vol_col not in tick_s.columns:
            continue

        vol_arr = tick_s[vol_col].values
        drops = vol_arr[:-1] - vol_arr[1:]      # drop[i] = tick[i] - tick[i+1]
        event_mask = drops >= threshold
        event_idx = np.where(event_mask)[0]     # indices into tick_s[1:] (i.e. tick i+1)

        if len(event_idx) == 0:
            continue

        # 对每个事件，统计 (t_prevs[i+1], tick_times[i+1]] 内对手方成交量
        # 用 searchsorted 向量化：找每个窗口左右边界在 real_times 中的位置
        t_events  = tick_times[event_idx + 1]
        tp_events = t_prevs[event_idx + 1]

        lo = np.searchsorted(real_times, tp_events, side='right')
        hi = np.searchsorted(real_times, t_events,  side='right')

        q80 = np.nanquantile(real_vol, 0.8) if len(real_vol) > 0 else np.inf
        counter_bs = 1 - bs_flag

        for k in range(len(event_idx)):
            win_vol = real_vol[lo[k]:hi[k]]
            win_bs  = real_bs[lo[k]:hi[k]]
            trade_vol  = win_vol[win_bs == counter_bs].sum()
            cancel_vol = max(0.0, drops[event_idx[k]] - trade_vol)
            records.append({
                'time':         t_events[k],
                't_prev':       tp_events[k],
                'side':         side,
                'total_drop':   drops[event_idx[k]],
                'trade_vol':    trade_vol,
                'cancel_vol':   cancel_vol,
                'large_cancel': cancel_vol >= q80,
            })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=['time', 't_prev', 'side', 'total_drop',
                 'trade_vol', 'cancel_vol', 'large_cancel']
    )
