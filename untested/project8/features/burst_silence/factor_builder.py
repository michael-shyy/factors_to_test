"""
factor_builder.py — 爆发-静默因子族汇总入口。

输出因子（~18个）：
  BURST_RATIO_buy/sell, BURST_ASYM,
  BURST_TYPE_PROG/SHOCK/DECAY_buy/sell, BURST_TYPE_PROG_ASYM,
  BURST_LARGE_RATIO,
  SILENCE_TYPE_RATIO,
  SILENCE_ORDER_ACTIVITY, SILENCE_CANCEL_RATIO, SILENCE_FULLCANCEL_RATE,
  BURST_WAIT_CV_buy/sell/diff
"""
import numpy as np
import sys
sys.path.insert(0, '/home/hysheng/project8')
sys.path.insert(0, '/home/hysheng')

from data.preprocessor import filter_continuous_auction
from .segmenter import segment_trades, split_by_direction, Segment
from .size_sequence import burst_type_ratios
from .silence_activity import silence_underwater


def build_burst_silence_factors(ctx) -> dict[str, float]:
    trade = filter_continuous_auction(
        ctx.trade[ctx.trade['TradeCode'] == 0].copy(), 'TradeTime'
    ).sort_values('TradeTime').reset_index(drop=True)
    order = ctx.order

    if len(trade) < 50:
        return {}

    ts    = trade['TradeTime'].values
    vols  = trade['TradeVolume'].values
    bs    = trade['BSFlag'].values.astype(int)

    q_large = np.nanquantile(vols, 0.8)
    q_small = np.nanquantile(vols, 0.2)

    # 全量段
    all_segs      = segment_trades(ts, vols, bs)
    bursts_all    = [s for s in all_segs if s.kind == 'burst']
    silences_all  = [s for s in all_segs if s.kind == 'silence']

    # 买/卖方向段
    buy_segs   = split_by_direction(ts, vols, bs, 0)
    sell_segs  = split_by_direction(ts, vols, bs, 1)
    buy_bursts  = [s for s in buy_segs  if s.kind == 'burst']
    sell_bursts = [s for s in sell_segs if s.kind == 'burst']
    buy_silence = [s for s in buy_segs  if s.kind == 'silence']
    sell_silence = [s for s in sell_segs if s.kind == 'silence']

    def _mean_dur(segs): return float(np.mean([s.end - s.start for s in segs])) if segs else np.nan
    def _d(a, b): return float(a - b) if not (np.isnan(a) or np.isnan(b)) else np.nan

    # BURST_RATIO = mean(L_burst) / mean(L_silence)
    burst_ratio_buy  = _safe_ratio(_mean_dur(buy_bursts),  _mean_dur(buy_silence))
    burst_ratio_sell = _safe_ratio(_mean_dur(sell_bursts), _mean_dur(sell_silence))
    burst_asym = _d(burst_ratio_buy, burst_ratio_sell)

    # 爆发内部量级结构
    types_buy  = burst_type_ratios(buy_bursts,  vols)
    types_sell = burst_type_ratios(sell_bursts, vols)
    prog_asym  = _d(types_buy['prog_ratio'], types_sell['prog_ratio'])

    # 大单占比
    burst_large_ratio = _burst_large_ratio(bursts_all, q_large)

    # 静默类型：大单成交后(消化型) vs 小单成交后(博弈型)
    silence_type_ratio = _silence_type_ratio(silences_all, vols, q_large, q_small)

    # 静默期水下活动
    underwater = silence_underwater(silences_all, order, ctx.trade)

    # 等待节拍稳定性（大单间隔 CV）
    wait_cv_buy  = _wait_cv(ts, vols, bs, q_large, 0)
    wait_cv_sell = _wait_cv(ts, vols, bs, q_large, 1)

    return {
        'BURST_RATIO_buy':          burst_ratio_buy,
        'BURST_RATIO_sell':         burst_ratio_sell,
        'BURST_ASYM':               burst_asym,
        'BURST_TYPE_PROG_buy':      types_buy['prog_ratio'],
        'BURST_TYPE_SHOCK_buy':     types_buy['shock_ratio'],
        'BURST_TYPE_DECAY_buy':     types_buy['decay_ratio'],
        'BURST_TYPE_PROG_sell':     types_sell['prog_ratio'],
        'BURST_TYPE_SHOCK_sell':    types_sell['shock_ratio'],
        'BURST_TYPE_DECAY_sell':    types_sell['decay_ratio'],
        'BURST_TYPE_PROG_ASYM':     prog_asym,
        'BURST_LARGE_RATIO':        burst_large_ratio,
        'SILENCE_TYPE_RATIO':       silence_type_ratio,
        'BURST_WAIT_CV_buy':        wait_cv_buy,
        'BURST_WAIT_CV_sell':       wait_cv_sell,
        'BURST_WAIT_CV_diff':       _d(wait_cv_buy, wait_cv_sell),
        **underwater,
    }


def _safe_ratio(a, b):
    if a is None or b is None or np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return float(a / b)


def _burst_large_ratio(bursts: list[Segment], q_large: float) -> float:
    if not bursts:
        return np.nan
    # 拼成一个大数组一次计算，避免逐段 loop
    all_vols = np.concatenate([seg.vol_seq for seg in bursts if len(seg.vol_seq) > 0])
    if len(all_vols) == 0:
        return np.nan
    return float((all_vols >= q_large).mean())


def _silence_type_ratio(silences: list[Segment], vols_all: np.ndarray,
                         q_large: float, q_small: float) -> float:
    """mean(消化型静默时长) / mean(博弈型静默时长)。"""
    digest, gamble = [], []
    for seg in silences:
        if len(seg.vol_seq) == 0:
            continue
        last_vol = seg.vol_seq[-1]
        dur = seg.end - seg.start
        if last_vol >= q_large:
            digest.append(dur)
        elif last_vol <= q_small:
            gamble.append(dur)
    m_d = float(np.mean(digest)) if digest else np.nan
    m_g = float(np.mean(gamble)) if gamble else np.nan
    return _safe_ratio(m_d, m_g)


def _wait_cv(ts, vols, bs, q_large, target_bs) -> float:
    """相邻两笔大单等待时间的变异系数。"""
    mask = (bs == target_bs) & (vols >= q_large)
    large_ts = ts[mask]
    if len(large_ts) < 3:
        return np.nan
    dt = np.diff(large_ts)
    m = np.mean(dt)
    return float(np.std(dt) / m) if m > 0 else np.nan
