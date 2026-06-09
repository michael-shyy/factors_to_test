"""
mlofi.py — 多档订单流失衡（MLOFI）。

MLOFI = sum_{i=1}^{10} w_i * (bid_vol_change_i - ask_vol_change_i)
权重 w_i = exp(-λ*(i-1))，远档指数衰减。

时间戳规则：基于 tick 快照差分，每个时刻 t 的 MLOFI 使用
tick[t].vol - tick[t-1].vol，不使用任何 t+1 之后的数据。
"""
import numpy as np
import pandas as pd


def compute_mlofi(
    tick: pd.DataFrame,
    n_levels: int = 10,
    decay: float = 0.5,
) -> pd.Series:
    """
    计算逐快照 MLOFI。

    Parameters
    ----------
    tick     : tick 快照 DataFrame（含 bdv1~10, akv1~10, time）
    n_levels : 档位数
    decay    : 指数衰减系数 λ

    Returns
    -------
    pd.Series，index=tick.index，第一行为 nan（差分无法计算）
    """
    tick = tick.sort_values('time').reset_index(drop=True)
    weights = np.exp(-decay * np.arange(n_levels))

    bid_cols = [f'bdv{i}' for i in range(1, n_levels + 1)]
    ask_cols = [f'akv{i}' for i in range(1, n_levels + 1)]

    bid_mat = tick[bid_cols].values.astype(float)
    ask_mat = tick[ask_cols].values.astype(float)

    dbid = np.diff(bid_mat, axis=0)  # (T-1, 10)
    dask = np.diff(ask_mat, axis=0)

    mlofi_raw = (dbid - dask) @ weights

    out = np.empty(len(tick))
    out[0] = np.nan
    out[1:] = mlofi_raw
    return pd.Series(out, index=tick.index, name='mlofi')


def mlofi_by_size(
    tick: pd.DataFrame,
    trade: pd.DataFrame,
    n_levels: int = 10,
    decay: float = 0.5,
    p_large: float = 0.8,
    p_small: float = 0.2,
) -> dict[str, pd.Series]:
    """
    将每个 tick 窗口内的大单/小单成交量分别统计，
    按比例拆分全量 MLOFI 得到 MLOFI_large / MLOFI_small。

    拆分逻辑：在每个 (t_prev, t] 窗口内，
    大单成交量占比 = large_vol / total_vol，乘以全量 MLOFI。

    时间戳规则：拆分权重来自同一窗口内已发生的成交，无前视。
    """
    real_trade = trade[trade['TradeCode'] == 0].copy()
    vol = real_trade['TradeVolume'].values
    q_large = np.nanquantile(vol, p_large)
    q_small = np.nanquantile(vol, p_small)
    real_trade = real_trade.sort_values('TradeTime').reset_index(drop=True)

    mlofi_all = compute_mlofi(tick, n_levels, decay)
    tick_s = tick.sort_values('time').reset_index(drop=True)
    tick_times = tick_s['time'].values
    t_prevs = np.empty(len(tick_times))
    t_prevs[0] = tick_times[0] - 1
    t_prevs[1:] = tick_times[:-1]

    trade_times = real_trade['TradeTime'].values
    trade_vols  = real_trade['TradeVolume'].values

    large_ratio = np.full(len(tick_times), np.nan)
    small_ratio = np.full(len(tick_times), np.nan)

    # Vectorized: use searchsorted to find trade indices in each (t_prev, t] window
    left_idx  = np.searchsorted(trade_times, t_prevs,    side='right')  # exclusive left
    right_idx = np.searchsorted(trade_times, tick_times, side='right')  # inclusive right

    for i in range(len(tick_times)):
        lo, hi = left_idx[i], right_idx[i]
        if lo >= hi:
            continue
        w_vol = trade_vols[lo:hi]
        total = w_vol.sum()
        if total == 0:
            continue
        large_ratio[i] = w_vol[w_vol >= q_large].sum() / total
        small_ratio[i] = w_vol[w_vol <= q_small].sum() / total

    m = mlofi_all.values
    return {
        'mlofi_all':   mlofi_all,
        'mlofi_large': pd.Series(m * large_ratio, index=tick_s.index, name='mlofi_large'),
        'mlofi_small': pd.Series(m * small_ratio, index=tick_s.index, name='mlofi_small'),
    }


def zscore_daily(series: pd.Series) -> pd.Series:
    """日内 Z-score 标准化（减均值除标准差）。"""
    m = series.mean()
    s = series.std()
    if s == 0 or np.isnan(s):
        return series * np.nan
    return (series - m) / s
