"""
factor_builder.py — 坍缩不对称因子族汇总入口。

输出因子（~20个）：
  COLLAPSE_ASYM, resupply_bid/ask,
  CANCEL_RATIO_bid/ask/ASYM, FULL/PART_CANCEL_RATIO_bid/ask,
  LARGE_CANCEL_SPEED_bid/ask, SHORT_CANCEL_RATIO_bid/ask,
  COMPETE_RATIO_bid/ask/ASYM, CV_bid/ask, COLLAPSE_CV_ASYM
"""
import sys
sys.path.insert(0, '/home/hysheng/project8')
sys.path.insert(0, '/home/hysheng')

from data.preprocessor import filter_continuous_auction
from .event_detector import detect_collapse_events
from .refill_tracker import compute_refill_speed
from .cancel_classifier import cancel_factors
from .cluster_stats import collapse_cv


def build_collapse_factors(ctx) -> dict[str, float]:
    """
    Parameters
    ----------
    ctx : DailyDataContext（需含 lifecycle 属性）

    Returns
    -------
    dict[str, float]
    """
    tick  = filter_continuous_auction(ctx.tick.copy(), 'time')
    trade = filter_continuous_auction(ctx.trade.copy(), 'TradeTime')
    order = ctx.order

    if len(tick) < 10:
        return {}

    events = detect_collapse_events(tick, trade, order)
    refill = compute_refill_speed(events, tick)

    lc = ctx.lifecycle.copy()
    # 从 geo_input 补充 is_large_order / is_short_order（lifecycle 本身不含）
    geo = ctx.geo_input
    tag_cols = [c for c in ('is_large_order', 'is_short_order') if c in geo.columns]
    if tag_cols:
        lc = lc.merge(geo[['sysid'] + tag_cols].drop_duplicates('sysid'),
                      on='sysid', how='left')
    cancel = cancel_factors(lc, tick) if len(lc) > 0 else {}

    cv = collapse_cv(events)

    return {**refill, **cancel, **cv}
