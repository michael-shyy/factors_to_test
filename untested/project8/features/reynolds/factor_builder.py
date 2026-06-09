"""
factor_builder.py — 雷诺数复合因子族汇总入口。

输出因子（18个）：
  Re_OF, Re_OF_raw, Re_buy, Re_sell, Re_asym,
  Re_large, Re_size_ratio, Re_probe, Re_probe_diff,
  PENETRATION_DEPTH_buy, PENETRATION_DEPTH_sell, PENETRATION_DEPTH_ASYM,
  PENETRATION_RE_buy, PENETRATION_RE_sell,
  IMPACT_EFF_trend, IMPACT_EFF_asym,
  IMPACT_DECAY_slope_buy, IMPACT_DECAY_slope_sell, IMPACT_DECAY_slope_diff
"""
import numpy as np
import sys
sys.path.insert(0, '/home/hysheng/project8')
sys.path.insert(0, '/home/hysheng')

from data.preprocessor import filter_continuous_auction
from .mlofi import compute_mlofi, mlofi_by_size, zscore_daily
from .resilience import compute_resilience
from .penetration_depth import compute_penetration_depth
from .impact_efficiency import compute_impact_efficiency
from .dead_zone import re_of_daily
from .probe_reynolds import probe_re_factor


def build_reynolds_factors(ctx) -> dict[str, float]:
    """
    计算单只股票单日所有 Re_* / PENETRATION_* / IMPACT_* 因子。

    Parameters
    ----------
    ctx : DailyDataContext

    Returns
    -------
    dict[str, float]
    """
    tick  = ctx.tick.sort_values('time').reset_index(drop=True)
    trade = filter_continuous_auction(
        ctx.trade[ctx.trade['TradeCode'] == 0].copy(), 'TradeTime'
    )
    order = ctx.order

    if len(trade) < 100:
        return {}

    # ── 1. MLOFI ────────────────────────────────────────────────────────
    tick_cont = filter_continuous_auction(tick.copy(), 'time')
    mlofi_dict = mlofi_by_size(tick_cont, trade)
    m_all   = zscore_daily(mlofi_dict['mlofi_all'].dropna())
    m_large = zscore_daily(mlofi_dict['mlofi_large'].dropna())
    m_small = zscore_daily(mlofi_dict['mlofi_small'].dropna())

    # ── 2. 恢复速度 ──────────────────────────────────────────────────────
    resil = compute_resilience(tick_cont, trade)
    resil_ask  = resil['resil_ask']
    resil_bid  = resil['resil_bid']
    resil_norm = resil['resil_norm']

    def _mn(s): return float(np.nanmean(s.values)) if len(s) > 0 else np.nan

    mn_all   = _mn(m_all)
    mn_large = _mn(zscore_daily(mlofi_dict['mlofi_large'].dropna()))
    mn_small = _mn(zscore_daily(mlofi_dict['mlofi_small'].dropna()))

    # 按方向拆分 MLOFI（取买方/卖方发起时刻的 MLOFI 均值）
    real = trade.copy()
    real_ts = real['TradeTime'].values
    real_bs = real['BSFlag'].values
    tick_ts = tick_cont['time'].values

    # 将 trade 时刻映射到最近的 tick Z-score MLOFI 值
    # 用 m_all 的值（已 Z-score），需要对齐 index
    m_all_reindexed = m_all.reindex(mlofi_dict['mlofi_all'].dropna().index).values
    mlofi_znorm_arr = zscore_daily(mlofi_dict['mlofi_all']).values
    idx = np.searchsorted(tick_ts, real_ts, side='right') - 1
    idx = np.clip(idx, 0, len(mlofi_znorm_arr) - 1)
    mlofi_at_trade = mlofi_znorm_arr[idx]

    mn_buy  = float(np.nanmean(mlofi_at_trade[real_bs == 0])) if (real_bs == 0).sum() > 0 else np.nan
    mn_sell = float(np.nanmean(mlofi_at_trade[real_bs == 1])) if (real_bs == 1).sum() > 0 else np.nan

    # ── 3. Re_OF 系列 ────────────────────────────────────────────────────
    re_base = re_of_daily(m_all.values, resil_norm)
    re_of     = re_base['Re_OF']
    re_of_raw = re_base['Re_OF_raw']

    re_buy  = _safe_mul(mn_buy,   resil_ask)
    re_sell = _safe_mul(mn_sell,  resil_bid)
    re_asym = _safe_diff(re_buy, re_sell)

    re_large      = _safe_mul(mn_large, resil_norm)
    re_size_ratio = (re_large / re_of if re_of and not np.isnan(re_of) and re_of != 0 else np.nan)

    # ── 4. 探价前置版 ────────────────────────────────────────────────────
    tick_e = ctx.tick_enriched if hasattr(ctx, 'tick_enriched') else tick_cont
    probe_res = probe_re_factor(
        ctx.trade, tick_e, m_all, resil_ask, resil_bid
    )

    # ── 5. 穿透深度 ──────────────────────────────────────────────────────
    pen = compute_penetration_depth(tick_cont, trade)

    # ── 6. 冲击效率 ──────────────────────────────────────────────────────
    imp = compute_impact_efficiency(tick_cont, trade)

    return {
        'Re_OF':                    re_of,
        'Re_OF_raw':                re_of_raw,
        'Re_buy':                   re_buy,
        'Re_sell':                  re_sell,
        'Re_asym':                  re_asym,
        'Re_large':                 re_large,
        'Re_size_ratio':            re_size_ratio,
        **probe_res,
        **pen,
        **imp,
    }


def _safe_mul(a, b) -> float:
    if a is None or b is None:
        return np.nan
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a * b)


def _safe_diff(a, b) -> float:
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a - b)
