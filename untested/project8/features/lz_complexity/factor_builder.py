"""
factor_builder.py — LZ76 族因子汇总入口。

输入：DailyDataContext（含 tick/order/trade）
输出：dict，键为因子名，值为日频标量

时间戳规则：所有子模块均使用 shift/rolling backward 确保无前视；
running VWAP 由 snapshot_aligner 提供，严禁使用全日累计 totalAmt/totalVolume。
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/hysheng/project8')
sys.path.insert(0, '/home/hysheng')

from data.preprocessor import filter_continuous_auction
from .encoder import encode_binary, encode_six_fast
from .lz76 import lz76_normalized, lz76_subseries, rolling_lz76
from .transition_matrix import transition_kl_asym
from .dict_stability import dict_stability_daily
from .price_context import price_context_masks, lz_price_context
from .regime_classifier import rolling_vol_label, split_by_regime
from .granger import granger_direction
from .probe_lz import probe_lz_factors


def _mean_last_hour(arr: np.ndarray, timestamps: np.ndarray) -> float:
    """取收盘前 1 小时（13:00–14:00）的均值作为日频因子值。"""
    local_sec = (timestamps + 28800) % 86400
    mask = (local_sec >= 13 * 3600) & (local_sec < 14 * 3600)
    vals = arr[mask]
    finite = vals[np.isfinite(vals)]
    return float(np.mean(finite)) if len(finite) > 0 else np.nan


def build_lz_factors(ctx) -> dict[str, float]:
    """
    计算单只股票单日所有 LZ_* 因子。

    Parameters
    ----------
    ctx : DailyDataContext

    Returns
    -------
    dict[str, float]  — 20–24 个 LZ_ 前缀因子
    """
    # 过滤连续竞价时段
    trade = filter_continuous_auction(ctx.trade[ctx.trade['TradeCode'] == 0].copy(), 'TradeTime')
    if len(trade) < 200:
        return {}

    trade = trade.sort_values('TradeTime').reset_index(drop=True)
    bs_flag = trade['BSFlag'].values.astype(np.int8)
    volume  = trade['TradeVolume'].values
    price   = trade['TradePrice'].values
    ts      = trade['TradeTime'].values

    # ── 编码 ────────────────────────────────────────────────────────────
    seq_bin = encode_binary(bs_flag)
    seq_six = encode_six_fast(bs_flag, volume)

    W2 = 500   # 二值窗口
    W6 = 800   # 六符号窗口
    STEP  = 10   # rolling step（均值统计用）
    STEP6 = 100  # 六符号 step

    # ── 基础 LZ ─────────────────────────────────────────────────────────
    roll_bin = rolling_lz76(seq_bin, W2, alphabet_size=2, step=STEP)
    roll_six = rolling_lz76(seq_six, W6, alphabet_size=6, step=STEP6)

    lz_total = _mean_last_hour(roll_bin, ts)
    lz_multi = _mean_last_hour(roll_six, ts)

    buy_mask  = bs_flag == 0
    sell_mask = bs_flag == 1
    q80 = np.nanquantile(volume, 0.8)
    q20 = np.nanquantile(volume, 0.2)
    large_mask = volume >= q80
    small_mask = volume <= q20

    # 子序列用价格涨跌方向重编码（1=上涨，0=下跌/持平）
    price_dir = (np.diff(price) > 0).astype(np.int8)
    def _subseq_lz(mask: np.ndarray) -> float:
        sub_mask = mask[1:]
        sub = price_dir[sub_mask]
        return lz76_normalized(sub[-W2:]) if len(sub) >= W2 // 4 else np.nan

    lz_buy   = _subseq_lz(buy_mask)
    lz_sell  = _subseq_lz(sell_mask)
    lz_large = _subseq_lz(large_mask)
    lz_small = _subseq_lz(small_mask)

    lz_diff       = _safe_diff(lz_buy, lz_sell)
    lz_size_diff  = _safe_diff(lz_large, lz_small)

    # ΔLZ：用 roll_bin（step=10）的两段均值，与 baseline 定义一致
    n_rb = len(roll_bin)
    if n_rb >= W2 * 2:
        lz_cur  = np.nanmean(roll_bin[n_rb - W2 // 2:])
        lz_prev = np.nanmean(roll_bin[n_rb - W2 - W2 // 2: n_rb - W2])
        delta_lz = float(lz_cur - lz_prev)
    else:
        delta_lz = np.nan

    # ── 价格背景条件化 LZ ────────────────────────────────────────────────
    snap = ctx.snap_features.sort_values('time').reset_index(drop=True)
    snap_prices = snap['running_vwap'].ffill().values
    snap_vol    = (snap['buy_trade_vol'] + snap['sell_trade_vol']).values

    # 将 trade 映射到最近的 tick 时刻，取对应 running_vwap 和 sigma
    mid = (ctx.tick.sort_values('time')['akp1'].values +
           ctx.tick.sort_values('time')['bdp1'].values) / 2.0
    tick_ts = ctx.tick.sort_values('time')['time'].values
    # 将 trade 时刻对应的 mid price（使用 ≤t 的最近 tick，无前视）
    idx = np.searchsorted(tick_ts, ts, side='right') - 1
    idx = np.clip(idx, 0, len(mid) - 1)
    trade_mid = mid[idx]
    trade_vol_cum = np.cumsum(volume)
    trade_pv_cum  = np.cumsum(price * volume)
    run_vwap = np.empty(len(price))
    run_vwap[0] = np.nan
    run_vwap[1:] = np.where(trade_vol_cum[:-1] > 0,
                             trade_pv_cum[:-1] / trade_vol_cum[:-1], np.nan)
    # rolling sigma（75 笔 ≈ 5min）
    run_sigma = _rolling_std_shift1(price, 75)

    ctx_masks = price_context_masks(price, volume)
    price_ctx = lz_price_context(seq_bin, ctx_masks, W2)

    # ── 转移矩阵 KL ──────────────────────────────────────────────────────
    kl_total, kl_buy, kl_sell = transition_kl_asym(
        seq_six[-W6:] if len(seq_six) >= W6 else seq_six, k=6
    )
    kl_asym = _safe_diff(kl_buy, kl_sell)

    # ── 词典时间稳定性 ────────────────────────────────────────────────────
    lz_dict_stability, lz_dict_volatility = dict_stability_daily(seq_six, ts)

    # ── 高/低波动分层 ─────────────────────────────────────────────────────
    vol_labels = rolling_vol_label(price, ts)
    hi_seq, lo_seq = split_by_regime(seq_bin, vol_labels)
    lz_highvol    = lz76_normalized(hi_seq[-W2:]) if len(hi_seq) >= W2 // 4 else np.nan
    lz_lowvol     = lz76_normalized(lo_seq[-W2:]) if len(lo_seq) >= W2 // 4 else np.nan
    lz_regime_diff = _safe_diff(lz_highvol, lz_lowvol)

    # ── Granger 领先方向 ──────────────────────────────────────────────────
    roll_buy  = rolling_lz76(seq_bin[buy_mask],  W2, step=STEP)
    roll_sell = rolling_lz76(seq_bin[sell_mask], W2, step=STEP)
    min_len = min(len(roll_buy), len(roll_sell))
    lz_granger = granger_direction(roll_buy[:min_len], roll_sell[:min_len])

    # ── 探价前置版 LZ ─────────────────────────────────────────────────────
    if hasattr(ctx, 'tick_enriched'):
        tick_e = ctx.tick_enriched.sort_values('time')
        tick_ts_e = tick_e['time'].values
        probe_raw = tick_e['probe_flag'].values
        idx2 = np.searchsorted(tick_ts_e, ts, side='right') - 1
        idx2 = np.clip(idx2, 0, len(probe_raw) - 1)
        probe_flag_per_trade = probe_raw[idx2].astype(np.int8)
    else:
        probe_flag_per_trade = np.zeros(len(seq_bin), dtype=np.int8)

    probe_res = probe_lz_factors(seq_bin, bs_flag, volume, probe_flag_per_trade, W2)

    # ── 汇总 ──────────────────────────────────────────────────────────────
    result = {
        'LZ_total':              lz_total,
        'LZ_multi':              lz_multi,
        'LZ_buy':                lz_buy,
        'LZ_sell':               lz_sell,
        'LZ_large':              lz_large,
        'LZ_small':              lz_small,
        'LZ_diff':               lz_diff,
        'LZ_size_diff':          lz_size_diff,
        'LZ_delta':              delta_lz,
        'LZ_upper':              price_ctx['LZ_upper'],
        'LZ_lower':              price_ctx['LZ_lower'],
        'LZ_price_context_diff': price_ctx['LZ_price_context_diff'],
        'LZ_vwap_regime':        price_ctx['LZ_vwap_regime'],
        'LZ_new_high_regime':    price_ctx['LZ_new_high_regime'],
        'LZ_syntax_kl':          kl_total,
        'LZ_syntax_kl_asym':     kl_asym,
        'LZ_dict_stability':     lz_dict_stability,
        'LZ_dict_volatility':    lz_dict_volatility,
        'LZ_highvol':            lz_highvol,
        'LZ_lowvol':             lz_lowvol,
        'LZ_regime_diff':        lz_regime_diff,
        'LZ_granger_dir':        float(lz_granger),
        **probe_res,
    }
    return result


# ── 工具函数 ───────────────────────────────────────────────────────────

def _safe_diff(a, b) -> float:
    if a is None or b is None:
        return np.nan
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a - b)


def _rolling_std_shift1(arr: np.ndarray, window: int) -> np.ndarray:
    """shift(1) rolling std，t 时刻用 [t-window, t) 历史。"""
    return pd.Series(arr).shift(1).rolling(window, min_periods=2).std().values
