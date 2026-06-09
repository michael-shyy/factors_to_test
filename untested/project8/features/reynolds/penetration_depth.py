"""
penetration_depth.py — 大单穿透档位数及弹性复合。

穿透档位数：一笔大单从开始成交到完成，连续吃掉的价格档位数。
通过比对成交前后 tick 的最优价变化推算。

时间戳规则：成交前 tick 取 <= TradeTime，成交后 tick 取 > TradeTime，无前视。
"""
import numpy as np
import pandas as pd
from .resilience import compute_resilience


def compute_penetration_depth(
    tick: pd.DataFrame,
    trade: pd.DataFrame,
    p_large: float = 0.8,
    tick_size: float = 0.01,
) -> dict[str, float]:
    """
    Returns
    -------
    dict 含：
      PENETRATION_DEPTH_buy   : 买方大单平均穿透档位数
      PENETRATION_DEPTH_sell  : 卖方大单平均穿透档位数
      PENETRATION_DEPTH_ASYM  : buy - sell
      PENETRATION_RE_buy      : 穿透深度 × 恢复速度（买方）
      PENETRATION_RE_sell     : 穿透深度 × 恢复速度（卖方）
    """
    real = trade[trade['TradeCode'] == 0].copy()
    real = real.sort_values('TradeTime').reset_index(drop=True)
    tick_s = tick.sort_values('time').reset_index(drop=True)

    vol = real['TradeVolume'].values
    q_large = np.nanquantile(vol, p_large)
    large_mask = vol >= q_large

    tick_times = tick_s['time'].values
    ask1 = tick_s['akp1'].values
    bid1 = tick_s['bdp1'].values
    trade_times = real['TradeTime'].values
    trade_bs    = real['BSFlag'].values

    depths_buy, depths_sell = [], []

    for idx in np.where(large_mask)[0]:
        t = trade_times[idx]
        bs = trade_bs[idx]

        ti_pre  = np.searchsorted(tick_times, t, side='right') - 1
        ti_post = np.searchsorted(tick_times, t, side='right')

        if ti_pre < 0 or ti_post >= len(tick_times):
            continue

        if bs == 0:  # 买方冲击：ask1 上移
            before = ask1[ti_pre]
            after  = ask1[ti_post]
            if before > 0 and after > before:
                depth = max(1, round((after - before) / tick_size))
            else:
                depth = 1
            depths_buy.append(depth)
        else:        # 卖方冲击：bid1 下移
            before = bid1[ti_pre]
            after  = bid1[ti_post]
            if before > 0 and after < before:
                depth = max(1, round((before - after) / tick_size))
            else:
                depth = 1
            depths_sell.append(depth)

    def _mean(lst):
        return float(np.mean(lst)) if lst else np.nan

    pd_buy  = _mean(depths_buy)
    pd_sell = _mean(depths_sell)
    pd_asym = _safe_diff(pd_buy, pd_sell)

    # 弹性复合
    resil = compute_resilience(tick, trade, p_large)
    re_buy  = (pd_buy  * resil['resil_ask'] if not np.isnan(pd_buy)  else np.nan)
    re_sell = (pd_sell * resil['resil_bid'] if not np.isnan(pd_sell) else np.nan)

    return {
        'PENETRATION_DEPTH_buy':  pd_buy,
        'PENETRATION_DEPTH_sell': pd_sell,
        'PENETRATION_DEPTH_ASYM': pd_asym,
        'PENETRATION_RE_buy':     re_buy,
        'PENETRATION_RE_sell':    re_sell,
    }


def _safe_diff(a, b):
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a - b)
