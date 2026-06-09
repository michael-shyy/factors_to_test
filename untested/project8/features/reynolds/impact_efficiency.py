"""
impact_efficiency.py — 冲击效率时变与连续冲击衰减模式。

时间戳规则：
- impact_eff 用大单成交后 N 个 tick（N=10）的中间价变化，tick 取 > TradeTime
- 衰减斜率拟合均为回望序列，无前视
"""
import numpy as np
import pandas as pd


def _mid(tick: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = tick.sort_values('time').reset_index(drop=True)
    return t['time'].values, (t['akp1'].values + t['bdp1'].values) / 2.0


def compute_impact_efficiency(
    tick: pd.DataFrame,
    trade: pd.DataFrame,
    p_large: float = 0.8,
    n_ticks_after: int = 10,
    burst_gap_sec: float = 30.0,
) -> dict[str, float]:
    """
    Returns
    -------
    IMPACT_EFF_trend    : 全天大单 impact_eff 日内线性趋势斜率
    IMPACT_EFF_asym     : mean(impact_eff_buy) / mean(impact_eff_sell)
    IMPACT_DECAY_slope_buy  : 买方连续冲击衰减斜率均值
    IMPACT_DECAY_slope_sell : 卖方连续冲击衰减斜率均值
    IMPACT_DECAY_slope_diff : buy - sell
    """
    real = trade[trade['TradeCode'] == 0].copy()
    real = real.sort_values('TradeTime').reset_index(drop=True)

    vol = real['TradeVolume'].values
    q_large = np.nanquantile(vol, p_large)
    large_mask = vol >= q_large
    large_idx = np.where(large_mask)[0]

    tick_times, mid = _mid(tick)
    trade_times = real['TradeTime'].values
    trade_bs    = real['BSFlag'].values
    trade_vol   = real['TradeVolume'].values

    # ── 1. 单笔冲击效率 ─────────────────────────────────────────────────
    eff_buy, eff_sell = [], []
    eff_order = []  # (i_large, eff) 用于趋势拟合

    for rank, idx in enumerate(large_idx):
        t = trade_times[idx]
        bs = trade_bs[idx]
        v  = trade_vol[idx]

        ti_pre = np.searchsorted(tick_times, t, side='right') - 1
        ti_post = ti_pre + n_ticks_after
        if ti_pre < 0 or ti_post >= len(tick_times):
            continue

        delta_mid = abs(mid[ti_post] - mid[ti_pre])
        eff = delta_mid / (v + 1e-8)
        eff_order.append((rank, eff, bs))

        if bs == 0:
            eff_buy.append(eff)
        else:
            eff_sell.append(eff)

    def _mean(lst):
        return float(np.nanmean(lst)) if lst else np.nan

    # 日内趋势斜率（线性回归 eff ~ rank）
    if len(eff_order) >= 5:
        ranks = np.array([x[0] for x in eff_order], dtype=float)
        effs  = np.array([x[1] for x in eff_order], dtype=float)
        valid = np.isfinite(effs)
        if valid.sum() >= 3:
            p = np.polyfit(ranks[valid], effs[valid], 1)
            trend = float(p[0])
        else:
            trend = np.nan
    else:
        trend = np.nan

    mean_buy  = _mean(eff_buy)
    mean_sell = _mean(eff_sell)
    asym = (mean_buy / mean_sell
            if (mean_sell and not np.isnan(mean_sell) and mean_sell > 0) else np.nan)

    # ── 2. 连续冲击衰减斜率 ────────────────────────────────────────────
    decay_buy, decay_sell = _burst_decay_slopes(
        trade_times, trade_bs, trade_vol, large_mask, tick_times, mid,
        burst_gap_sec, n_ticks_after
    )

    return {
        'IMPACT_EFF_trend':         trend,
        'IMPACT_EFF_asym':          asym,
        'IMPACT_DECAY_slope_buy':   _mean(decay_buy),
        'IMPACT_DECAY_slope_sell':  _mean(decay_sell),
        'IMPACT_DECAY_slope_diff':  _safe_diff(_mean(decay_buy), _mean(decay_sell)),
    }


def _burst_decay_slopes(
    trade_times, trade_bs, trade_vol, large_mask,
    tick_times, mid, burst_gap_sec, n_ticks_after,
) -> tuple[list, list]:
    """识别连续大单序列，对每个序列拟合价格冲击衰减斜率。"""
    large_idx = np.where(large_mask)[0]
    if len(large_idx) < 2:
        return [], []

    slopes_buy, slopes_sell = [], []
    cluster: list[int] = [large_idx[0]]

    def _fit_cluster(c):
        impacts = []
        for i in c:
            t = trade_times[i]
            ti = np.searchsorted(tick_times, t, side='right') - 1
            ti_post = ti + n_ticks_after
            if ti < 0 or ti_post >= len(tick_times):
                impacts.append(np.nan)
                continue
            impacts.append(abs(mid[ti_post] - mid[ti]))
        impacts = np.array(impacts, dtype=float)
        valid = np.isfinite(impacts)
        if valid.sum() < 3:
            return np.nan
        x = np.arange(len(impacts), dtype=float)[valid]
        p = np.polyfit(x, impacts[valid], 1)
        return float(p[0])

    for idx in large_idx[1:]:
        prev = cluster[-1]
        if trade_times[idx] - trade_times[prev] <= burst_gap_sec:
            cluster.append(idx)
        else:
            if len(cluster) >= 3:
                bs = trade_bs[cluster[0]]
                slope = _fit_cluster(cluster)
                if not np.isnan(slope):
                    (slopes_buy if bs == 0 else slopes_sell).append(slope)
            cluster = [idx]

    if len(cluster) >= 3:
        bs = trade_bs[cluster[0]]
        slope = _fit_cluster(cluster)
        if not np.isnan(slope):
            (slopes_buy if bs == 0 else slopes_sell).append(slope)

    return slopes_buy, slopes_sell


def _safe_diff(a, b):
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a - b)
