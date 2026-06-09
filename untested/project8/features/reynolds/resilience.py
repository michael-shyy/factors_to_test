"""
resilience.py — 价格弹性恢复速度。

对每笔大单成交事件：
  1. 记录成交前最近 tick 的中间价 mid_pre
  2. 之后逐 tick 监测，记录首次回归到 mid_pre ± 0.5 tick 所用时间 T_recover
  3. resil_speed = 1 / T_recover（超时截断为 1/T_window）

分买方冲击（BSFlag=0）和卖方冲击（BSFlag=1）分别统计。

时间戳规则：mid_pre 取大单成交时刻之前（<= TradeTime）的最近 tick，
恢复监测向后追踪（> TradeTime），全程无前视。
"""
import numpy as np
import pandas as pd


def _mid(tick: pd.DataFrame) -> np.ndarray:
    return (tick['akp1'].values + tick['bdp1'].values) / 2.0


def compute_resilience(
    tick: pd.DataFrame,
    trade: pd.DataFrame,
    p_large: float = 0.8,
    recovery_window_sec: float = 30.0,
    half_tick: float = 0.01,
) -> dict[str, float]:
    """
    计算全天大单冲击后的平均恢复速度。

    Parameters
    ----------
    tick               : tick 快照
    trade              : 逐笔成交
    p_large            : 大单阈值分位数
    recovery_window_sec: 超时截断时间（秒）
    half_tick          : 恢复判定阈值（价格单位）

    Returns
    -------
    dict 含 resil_ask（买方冲击后卖侧恢复速度均值）、
            resil_bid（卖方冲击后买侧恢复速度均值）、
            resil_norm（全量恢复速度 Z-score 均值）
    """
    real_trade = trade[trade['TradeCode'] == 0].copy()
    real_trade = real_trade.sort_values('TradeTime').reset_index(drop=True)
    tick_s = tick.sort_values('time').reset_index(drop=True)

    vol = real_trade['TradeVolume'].values
    q_large = np.nanquantile(vol, p_large)
    large_mask = vol >= q_large

    mid = _mid(tick_s)
    tick_times = tick_s['time'].values

    trade_times = real_trade['TradeTime'].values
    trade_bs    = real_trade['BSFlag'].values

    t_fill = 1.0 / recovery_window_sec

    speeds_buy, speeds_sell = [], []

    for idx in np.where(large_mask)[0]:
        t_trade = trade_times[idx]
        bs      = trade_bs[idx]

        # mid_pre：大单成交时刻之前（<=）最近 tick
        ti = np.searchsorted(tick_times, t_trade, side='right') - 1
        if ti < 0:
            continue
        mid_pre = mid[ti]
        if np.isnan(mid_pre) or mid_pre <= 0:
            continue

        # 向后追踪：用 searchsorted 确定窗口右边界，再找首个恢复 tick
        t_end = t_trade + recovery_window_sec
        tj_end = np.searchsorted(tick_times, t_end, side='right')
        recovered = False
        for tj in range(ti + 1, tj_end):
            if abs(mid[tj] - mid_pre) <= half_tick:
                dt = tick_times[tj] - t_trade
                speed = 1.0 / max(dt, 1e-3)
                recovered = True
                if bs == 0:
                    speeds_buy.append(speed)
                else:
                    speeds_sell.append(speed)
                break
        if not recovered:
            if bs == 0:
                speeds_buy.append(t_fill)
            else:
                speeds_sell.append(t_fill)

    def _mean(lst):
        return float(np.mean(lst)) if lst else np.nan

    resil_ask = _mean(speeds_buy)   # 买方冲击 → 卖侧填补速度
    resil_bid = _mean(speeds_sell)  # 卖方冲击 → 买侧填补速度
    all_speeds = speeds_buy + speeds_sell
    resil_norm = _mean(all_speeds)

    return {
        'resil_ask':  resil_ask,
        'resil_bid':  resil_bid,
        'resil_norm': resil_norm,
    }
