"""
factor_builder.py — 盘口互信息因子族汇总入口。

输出因子（~15个）：
  MI_level1, MI_level1_3, MI_trend,
  MI_quote_driven, MI_trade_driven, MI_source_ratio,
  MI_CURVE_PEAK, MI_CURVE_SLOPE, MI_CURVE_FLATNESS,
  CMI_level1,
  TE_bid2ask, TE_ask2bid, TE_DIR,
  BOB_SYMM
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/hysheng/project8')
sys.path.insert(0, '/home/hysheng')

from data.preprocessor import filter_continuous_auction
from .ksg_estimator import ksg_mi, rolling_ksg_mi
from .transfer_entropy import transfer_entropy


def build_mi_factors(ctx) -> dict[str, float]:
    tick  = filter_continuous_auction(ctx.tick.copy(), 'time').sort_values('time').reset_index(drop=True)
    trade = filter_continuous_auction(ctx.trade[ctx.trade['TradeCode'] == 0].copy(), 'TradeTime')

    if len(tick) < 100:
        return {}

    bdv1 = tick['bdv1'].values.astype(float)
    akv1 = tick['akv1'].values.astype(float)
    tick_times = tick['time'].values

    # 差分 + 微小扰动（处理零膨胀）
    rng = np.random.default_rng(0)
    dbid = np.diff(bdv1);  dbid = dbid + rng.normal(0, 0.1, len(dbid))
    dask = np.diff(akv1);  dask = dask + rng.normal(0, 0.1, len(dask))
    ts_diff = tick_times[1:]

    W = min(500, len(dbid) // 2)
    if W < 50:
        return {}

    # ── 基础 MI ──────────────────────────────────────────────────────────
    valid = np.isfinite(dbid) & np.isfinite(dask)
    mi_l1 = ksg_mi(dbid[valid], dask[valid]) if valid.sum() >= 10 else np.nan

    # 前3档均值
    def _mean_3(prefix):
        cols = [f'{prefix}{i}' for i in range(1, 4) if f'{prefix}{i}' in tick.columns]
        if not cols:
            return np.full(len(tick), np.nan)
        return tick[cols].mean(axis=1).values

    bdv1_3 = _mean_3('bdv')
    akv1_3 = _mean_3('akv')
    db3 = np.diff(bdv1_3) + rng.normal(0, 0.1, len(bdv1_3) - 1)
    da3 = np.diff(akv1_3) + rng.normal(0, 0.1, len(akv1_3) - 1)
    v3  = np.isfinite(db3) & np.isfinite(da3)
    mi_l1_3 = ksg_mi(db3[v3], da3[v3]) if v3.sum() >= 10 else np.nan

    # MI 趋势（rolling MI 线性斜率）
    roll_mi = rolling_ksg_mi(dbid, dask, window=W, step=10)
    valid_roll = np.isfinite(roll_mi)
    if valid_roll.sum() >= 5:
        x = np.where(valid_roll)[0].astype(float)
        y = roll_mi[valid_roll]
        mi_trend = float(np.polyfit(x, y, 1)[0])
    else:
        mi_trend = np.nan

    # ── 成交驱动 vs 主动调整的 MI 拆分 ──────────────────────────────────
    trade_times = trade['TradeTime'].values
    is_trade_tick = np.zeros(len(ts_diff), dtype=bool)
    if len(trade_times) > 0:
        # 向量化：tick 区间 (tick_times[i], tick_times[i+1]] 内有成交则标记
        # ts_diff[i] 对应区间 (tick_times[i], tick_times[i+1]]
        # 用 unique 的 searchsorted 避免O(n*m)循环
        ti = np.searchsorted(tick_times[1:], trade_times, side='left')
        ti = np.clip(ti, 0, len(ts_diff) - 1)
        is_trade_tick[ti] = True

    v_trade  = np.isfinite(dbid) & np.isfinite(dask) & is_trade_tick
    v_quote  = np.isfinite(dbid) & np.isfinite(dask) & ~is_trade_tick
    mi_trade = ksg_mi(dbid[v_trade], dask[v_trade]) if v_trade.sum() >= 10 else np.nan
    mi_quote = ksg_mi(dbid[v_quote], dask[v_quote]) if v_quote.sum() >= 10 else np.nan
    mi_source_ratio = (_safe_ratio(mi_quote, mi_trade))

    # ── MI 档位曲线 ──────────────────────────────────────────────────────
    mi_curve = []
    for i in range(1, 11):
        bc = f'bdv{i}'; ac = f'akv{i}'
        if bc not in tick.columns or ac not in tick.columns:
            mi_curve.append(np.nan)
            continue
        db_i = np.diff(tick[bc].values.astype(float))
        da_i = np.diff(tick[ac].values.astype(float))
        vv = np.isfinite(db_i) & np.isfinite(da_i)
        mi_curve.append(ksg_mi(db_i[vv], da_i[vv]) if vv.sum() >= 10 else np.nan)

    mi_arr = np.array(mi_curve)
    valid_c = np.isfinite(mi_arr)
    if valid_c.sum() >= 3:
        peak_lv = int(np.nanargmax(mi_arr)) + 1
        x_lv = np.where(valid_c)[0].astype(float)
        curve_slope = float(np.polyfit(x_lv, mi_arr[valid_c], 1)[0])
        curve_flat  = float(1.0 / (np.nanstd(mi_arr) + 1e-8))
    else:
        peak_lv, curve_slope, curve_flat = np.nan, np.nan, np.nan

    # ── 条件 MI（控制 mid 价格漂移）────────────────────────────────────
    if 'akp1' in tick.columns and 'bdp1' in tick.columns:
        mid = (tick['akp1'].values + tick['bdp1'].values) / 2.0
        dmid = np.diff(mid)
        v_cmi = np.isfinite(dbid) & np.isfinite(dask) & np.isfinite(dmid)
        # 简化 CMI：先对 dbid/dask 分别对 dmid 做线性残差，再算 MI
        if v_cmi.sum() >= 15:
            dbid_r = _residual(dbid[v_cmi], dmid[v_cmi])
            dask_r = _residual(dask[v_cmi], dmid[v_cmi])
            cmi_l1 = ksg_mi(dbid_r, dask_r)
        else:
            cmi_l1 = np.nan
    else:
        cmi_l1 = np.nan

    # ── Transfer Entropy ──────────────────────────────────────────────
    te_b2a, te_a2b, te_dir = transfer_entropy(dbid, dask)

    # ── BOB_SYMM ─────────────────────────────────────────────────────
    bob_symm = _bob_symm(bdv1, akv1, W)

    return {
        'MI_level1':        mi_l1,
        'MI_level1_3':      mi_l1_3,
        'MI_trend':         mi_trend,
        'MI_quote_driven':  mi_quote,
        'MI_trade_driven':  mi_trade,
        'MI_source_ratio':  mi_source_ratio,
        'MI_CURVE_PEAK':    float(peak_lv) if not np.isnan(peak_lv) else np.nan,
        'MI_CURVE_SLOPE':   curve_slope,
        'MI_CURVE_FLATNESS': curve_flat,
        'CMI_level1':       cmi_l1,
        'TE_bid2ask':       te_b2a,
        'TE_ask2bid':       te_a2b,
        'TE_DIR':           te_dir,
        'BOB_SYMM':         bob_symm,
    }


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """简单 OLS 残差：y - (a*x + b)。"""
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 3:
        return y
    p = np.polyfit(x[valid], y[valid], 1)
    return y - np.polyval(p, x)


def _bob_symm(bdv1: np.ndarray, akv1: np.ndarray, window: int) -> float:
    """滚动 Pearson 相关系数均值（末尾 window 个快照）。"""
    n = min(len(bdv1), window)
    b = bdv1[-n:]
    a = akv1[-n:]
    valid = np.isfinite(b) & np.isfinite(a)
    if valid.sum() < 10:
        return np.nan
    return float(np.corrcoef(b[valid], a[valid])[0, 1])


def _safe_ratio(a, b):
    if np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return float(a / b)
