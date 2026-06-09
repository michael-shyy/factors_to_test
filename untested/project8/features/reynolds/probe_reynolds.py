"""
probe_reynolds.py — 探价前置版雷诺数。

对探价前置型大单单独计算 MLOFI × 恢复速度，与全量 Re_OF 作差。

时间戳规则：probe_flag 由预处理阶段（tick 粒度）计算，
映射到大单成交时刻时取 <= TradeTime 的最近 tick 标记，无前视。
"""
import numpy as np
import pandas as pd


def probe_re_factor(
    trade: pd.DataFrame,
    tick: pd.DataFrame,
    mlofi_norm: pd.Series,
    resil_ask: float,
    resil_bid: float,
    p_large: float = 0.8,
) -> dict[str, float]:
    """
    Parameters
    ----------
    trade      : 逐笔成交（含 probe_flag 列，若无则返回 nan）
    tick       : tick 快照（含 probe_flag 列）
    mlofi_norm : Z-score 标准化后的 MLOFI 时序（index 对应 tick.index）
    resil_ask  : 买方冲击恢复速度
    resil_bid  : 卖方冲击恢复速度

    Returns
    -------
    dict 含 Re_probe, Re_probe_diff
    """
    # 从 tick 的 probe_flag 映射到成交时刻
    if 'probe_flag' not in tick.columns:
        return {'Re_probe': np.nan, 'Re_probe_diff': np.nan}

    real = trade[trade['TradeCode'] == 0].copy()
    real = real.sort_values('TradeTime').reset_index(drop=True)
    tick_s = tick.sort_values('time').reset_index(drop=True)

    vol = real['TradeVolume'].values
    q_large = np.nanquantile(vol, p_large)
    large_mask = vol >= q_large

    tick_times  = tick_s['time'].values
    probe_flags = tick_s['probe_flag'].values
    trade_times = real['TradeTime'].values

    # 大单对应的 probe_flag（取最近 <= TradeTime 的 tick）
    idx = np.searchsorted(tick_times, trade_times[large_mask], side='right') - 1
    idx = np.clip(idx, 0, len(probe_flags) - 1)
    pf = probe_flags[idx]

    # 探价型大单对应的 tick 时刻位置
    large_trade_t = trade_times[large_mask]
    probe_trade_t = large_trade_t[pf == 1]

    if len(probe_trade_t) == 0:
        return {'Re_probe': np.nan, 'Re_probe_diff': np.nan}

    # 取探价型大单对应 tick 时刻的 MLOFI_norm 值均值
    ti_probe = np.searchsorted(tick_times, probe_trade_t, side='right') - 1
    ti_probe = np.clip(ti_probe, 0, len(mlofi_norm) - 1)
    mn_vals  = mlofi_norm.iloc[ti_probe].values
    mn_probe = float(np.nanmean(mn_vals))

    resil_mean = np.nanmean([resil_ask, resil_bid])
    re_probe = mn_probe * resil_mean if not np.isnan(resil_mean) else np.nan

    # Re_probe_diff = Re_probe - Re_OF_raw (用全量 MLOFI 均值作为基准)
    mn_all = float(np.nanmean(mlofi_norm.values))
    re_all = mn_all * resil_mean if not np.isnan(resil_mean) else np.nan
    diff = _safe_diff(re_probe, re_all)

    return {'Re_probe': re_probe, 'Re_probe_diff': diff}


def _safe_diff(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a - b)
