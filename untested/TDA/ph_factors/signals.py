"""
signals.py
==========
从 h5 原始数据计算 TDA 所需的全部主信号和环境维。

撤单识别依赖 DataProcessor.build_order_lifecycle，自动处理 SSE/SZSE 不对称：
  - 上交所：撤单在 order 表（OrderType==5）
  - 深交所：撤单在 trade 表（TradeCode==4）

数据源列名
----------
order_df : OrderTime(UTC秒), sysid, OrderPrice, OrderVolume, BSFlag(0=买/1=卖), OrderType
tick_df  : time(UTC秒), bdp1-5/bdv1-5, akp1-5/akv1-5
trade_df : TradeTime, sysid, BSFlag, TradePrice, TradeVolume, TradeCode, BuyOrderID, SellOrderID

输出
----
dict[str, pd.Series]，index 为 pd.DatetimeIndex（Asia/Shanghai），3 秒频率，FULL 时段。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# ── 时段常量 ──────────────────────────────────────────────────────────────────
_TZ_OFFSET  = 28800
_OPEN_S     = 9 * 3600 + 30 * 60   # 09:30
_CLOSE_S    = 15 * 3600             # 15:00
_LUNCH_S    = 11 * 3600 + 30 * 60  # 11:30
_LUNCH_E    = 13 * 3600             # 13:00
_OPEN_END_S = 10 * 3600 + 30 * 60  # 10:30
_BUCKET_SEC = 3


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _local_sec(utc_sec: np.ndarray) -> np.ndarray:
    return (utc_sec + _TZ_OFFSET) % 86400


def _trading_mask(local_sec: np.ndarray, segment: str = "FULL") -> np.ndarray:
    if segment == "FULL":
        return (
            (local_sec >= _OPEN_S) & (local_sec < _CLOSE_S)
            & ~((local_sec >= _LUNCH_S) & (local_sec < _LUNCH_E))
        )
    elif segment == "OPEN":
        return (local_sec >= _OPEN_S) & (local_sec < _OPEN_END_S)
    raise ValueError(f"Unknown segment: {segment}")


def _bucket_key(utc_sec: np.ndarray) -> np.ndarray:
    return (utc_sec // _BUCKET_SEC).astype(np.int64)


def _make_bucket_index(date_str: str) -> pd.DatetimeIndex:
    """生成 FULL 时段的 3 秒桶 DatetimeIndex（Asia/Shanghai）。"""
    base = pd.Timestamp(date_str, tz="Asia/Shanghai")
    am = pd.date_range(
        base + pd.Timedelta(hours=9, minutes=30),
        base + pd.Timedelta(hours=11, minutes=30),
        freq="3s", inclusive="left",
    )
    pm = pd.date_range(
        base + pd.Timedelta(hours=13),
        base + pd.Timedelta(hours=15),
        freq="3s", inclusive="left",
    )
    return am.append(pm)


# ── OFI_NET（tick 快照差分，5 档加权）────────────────────────────────────────

_OFI_WEIGHTS = np.array([1.0, 0.8, 0.6, 0.4, 0.2])


def _compute_ofi_from_tick(tick_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    df = tick_df[_trading_mask(_local_sec(tick_df["time"].values))].sort_values("time").reset_index(drop=True)
    if len(df) < 2:
        empty = pd.Series(dtype=np.float64)
        return empty, empty, empty

    ofi_buy = np.zeros(len(df), dtype=np.float64)
    ofi_sell = np.zeros(len(df), dtype=np.float64)

    for i, w in enumerate(_OFI_WEIGHTS, 1):
        bp = df[f"bdp{i}"].values; bv = df[f"bdv{i}"].values
        ap = df[f"akp{i}"].values; av = df[f"akv{i}"].values
        bp_p = np.r_[bp[0:1], bp[:-1]]; bv_p = np.r_[bv[0:1], bv[:-1]]
        ap_p = np.r_[ap[0:1], ap[:-1]]; av_p = np.r_[av[0:1], av[:-1]]

        ofi_buy  += w * np.where(bp > bp_p, bv,  np.where(bp == bp_p, bv - bv_p, -bv_p))
        ofi_sell += w * np.where(ap < ap_p, av,  np.where(ap == ap_p, av - av_p, -av_p))

    buckets = _bucket_key(df["time"].values)
    idx = pd.Index(buckets, name="bucket")
    s_buy = pd.Series(ofi_buy,  index=idx).groupby(level=0).sum()
    s_sell = pd.Series(ofi_sell, index=idx).groupby(level=0).sum()
    return s_buy - s_sell, s_buy, s_sell


# ── OB_IMBAL（tick 快照）─────────────────────────────────────────────────────

def _compute_ob_imbal(tick_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    df = tick_df[_trading_mask(_local_sec(tick_df["time"].values))].sort_values("time").reset_index(drop=True)
    v_bid = sum(df[f"bdv{i}"].values for i in range(1, 6))
    v_ask = sum(df[f"akv{i}"].values for i in range(1, 6))
    total = v_bid + v_ask + 1e-9
    buckets = _bucket_key(df["time"].values)
    idx = pd.Index(buckets, name="bucket")
    last = lambda a: pd.Series(a, index=idx).groupby(level=0).last()  # noqa
    return last((v_bid - v_ask) / total), last(v_bid / total), last(v_ask / total)


# ── DP_MICRO（tick 中间价差分）───────────────────────────────────────────────

def _compute_dp_micro(tick_df: pd.DataFrame) -> pd.Series:
    df = tick_df[_trading_mask(_local_sec(tick_df["time"].values))].sort_values("time").reset_index(drop=True)
    mid = (df["bdp1"].values + df["akp1"].values) / 2.0
    dp = np.r_[0.0, np.diff(mid)]
    buckets = _bucket_key(df["time"].values)
    return pd.Series(dp, index=pd.Index(buckets, name="bucket")).groupby(level=0).last()


# ── 撤单信号（复用 DataProcessor.build_order_lifecycle）─────────────────────
#
# lifecycle 输出含：sysid, OrderTime, BSFlag, OrderVolume,
#                   cancel_time(NaN=未撤), order_type_tag, filled_vol
#
# CANCEL_NET  : 3 秒桶内 (买撤笔数 - 卖撤笔数)
# CANCEL_RATE : 3 秒桶内 撤单笔数 / (挂单笔数 + 撤单笔数 + 1e-9)
# LIFE_TIME   : 3 秒桶内已撤委托的中位存活时间（cancel_time - OrderTime）

def _compute_cancel_signals(
    lifecycle: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    从 lifecycle 表提取撤单相关信号。

    lifecycle 必须已过滤至交易时段（build_order_lifecycle 内部已做）。

    Returns: CANCEL_NET, CANCEL_NET_BUY, CANCEL_NET_SELL, CANCEL_RATE, LIFE_TIME
    """
    lc = lifecycle.copy()
    ls = _local_sec(lc["OrderTime"].values)
    lc = lc[_trading_mask(ls)]

    # 有撤单记录的行
    cancelled = lc[lc["cancel_time"].notna()].copy()

    if len(cancelled) == 0:
        empty = pd.Series(dtype=np.float64)
        return empty, empty, empty, empty, empty

    # 撤单时间所在桶
    cancel_buckets = _bucket_key(cancelled["cancel_time"].values)
    is_buy  = (cancelled["BSFlag"].values == 0).astype(np.float64)
    is_sell = (cancelled["BSFlag"].values == 1).astype(np.float64)

    idx = pd.Index(cancel_buckets, name="bucket")
    s_buy  = pd.Series(is_buy,  index=idx).groupby(level=0).sum()
    s_sell = pd.Series(is_sell, index=idx).groupby(level=0).sum()
    cancel_net = s_buy - s_sell

    # CANCEL_RATE：按挂单时间所在桶统计
    order_buckets = _bucket_key(lc["OrderTime"].values)
    order_idx = pd.Index(order_buckets, name="bucket")
    is_cancel_flag = lc["cancel_time"].notna().astype(np.float64).values
    cancel_cnt = pd.Series(is_cancel_flag,                    index=order_idx).groupby(level=0).sum()
    order_cnt  = pd.Series(np.ones(len(lc), dtype=np.float64), index=order_idx).groupby(level=0).sum()
    cancel_rate = cancel_cnt / (order_cnt + 1e-9)

    # LIFE_TIME：存活时间 = cancel_time - OrderTime，按撤单桶归属
    cancelled["life"] = (cancelled["cancel_time"] - cancelled["OrderTime"]).clip(lower=0)
    life_time = (
        pd.Series(cancelled["life"].values, index=idx)
        .groupby(level=0).median()
    )

    return cancel_net, s_buy, s_sell, cancel_rate, life_time


# ── 环境维（tick）────────────────────────────────────────────────────────────

def _compute_spread(tick_df: pd.DataFrame) -> pd.Series:
    df = tick_df[_trading_mask(_local_sec(tick_df["time"].values))].sort_values("time").reset_index(drop=True)
    mid = (df["bdp1"].values + df["akp1"].values) / 2.0
    spread = (df["akp1"].values - df["bdp1"].values) / np.where(mid == 0, np.nan, mid)
    buckets = _bucket_key(df["time"].values)
    return pd.Series(spread, index=pd.Index(buckets, name="bucket")).groupby(level=0).last()


def _compute_depth5(tick_df: pd.DataFrame) -> pd.Series:
    df = tick_df[_trading_mask(_local_sec(tick_df["time"].values))].sort_values("time").reset_index(drop=True)
    depth = sum(df[f"bdv{i}"].values + df[f"akv{i}"].values for i in range(1, 6))
    buckets = _bucket_key(df["time"].values)
    return pd.Series(depth, index=pd.Index(buckets, name="bucket")).groupby(level=0).last()


def _compute_rv_proxy(dp_micro: pd.Series, window: int = 10) -> pd.Series:
    return dp_micro.pow(2).rolling(window, min_periods=1).sum()


# ── 标准化 ────────────────────────────────────────────────────────────────────

def zscore(series: pd.Series, mean: float, std: float) -> pd.Series:
    if std == 0 or np.isnan(std):
        return series * np.nan
    return (series - mean) / std


# ── 主入口 ────────────────────────────────────────────────────────────────────

def compute_all_signals(
    order_df: pd.DataFrame,
    tick_df:  pd.DataFrame,
    date_str: str,
    trade_df: Optional[pd.DataFrame] = None,
    stock_id: str = "",
    norm_params: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, pd.Series]:
    """
    计算全部主信号和环境维，对齐到 FULL 时段的 3 秒桶 DatetimeIndex。

    Args:
        order_df:    h5 order/table
        tick_df:     h5 tick/table
        date_str:    'YYYYMMDD'
        trade_df:    h5 trade/table（用于撤单识别，深交所必须传入）
        stock_id:    股票代码，如 'sh600000'（用于 SSE/SZSE 判断）
        norm_params: {signal_name: {'mean': float, 'std': float}}，None 则不标准化

    Returns:
        dict[str, pd.Series]，index 为 pd.DatetimeIndex（Asia/Shanghai）
    """
    from Processor.processor import DataProcessor

    full_idx = _make_bucket_index(date_str)
    bucket_to_dt = {int(ts.timestamp()) // _BUCKET_SEC: ts for ts in full_idx}

    def _reindex(s: pd.Series, fill: float = 0.0) -> pd.Series:
        if len(s) == 0:
            return pd.Series(fill, index=full_idx, dtype=np.float64)
        valid_keys = [k for k in s.index if k in bucket_to_dt]
        aligned = s[valid_keys].copy()
        aligned.index = pd.DatetimeIndex(
            [bucket_to_dt[k] for k in valid_keys], tz="Asia/Shanghai"
        )
        return aligned.reindex(full_idx, fill_value=fill)

    # ── tick 信号 ──
    ofi_net, ofi_buy, ofi_sell = _compute_ofi_from_tick(tick_df)
    ob_imbal, ob_buy, ob_sell  = _compute_ob_imbal(tick_df)
    dp_micro                   = _compute_dp_micro(tick_df)
    spread                     = _compute_spread(tick_df)
    depth5                     = _compute_depth5(tick_df)
    rv_proxy                   = _compute_rv_proxy(_reindex(dp_micro, 0.0))

    # ── 撤单信号：通过 lifecycle 统一处理 SSE/SZSE ──
    if trade_df is not None:
        lifecycle = DataProcessor.build_order_lifecycle(order_df, trade_df, stock_id=stock_id)
        cancel_net, c_buy, c_sell, cancel_rate, life_time = _compute_cancel_signals(lifecycle)
    else:
        # 无 trade_df 时退化为仅上交所逻辑（OrderType==5）
        lc = order_df.copy()
        ls = _local_sec(lc["OrderTime"].values)
        lc = lc[_trading_mask(ls)].copy()
        cancelled = lc[lc.get("OrderType", pd.Series(dtype=float)) == 5] if "OrderType" in lc.columns else lc.iloc[0:0]
        if len(cancelled) > 0:
            buckets = _bucket_key(cancelled["OrderTime"].values)
            idx = pd.Index(buckets, name="bucket")
            is_buy  = (cancelled["BSFlag"].values == 0).astype(np.float64)
            is_sell = (cancelled["BSFlag"].values == 1).astype(np.float64)
            c_buy   = pd.Series(is_buy,  index=idx).groupby(level=0).sum()
            c_sell  = pd.Series(is_sell, index=idx).groupby(level=0).sum()
            cancel_net = c_buy - c_sell
        else:
            cancel_net = c_buy = c_sell = pd.Series(dtype=np.float64)
        cancel_rate = life_time = pd.Series(dtype=np.float64)

    raw = {
        "OFI_NET":         _reindex(ofi_net,     0.0),
        "OFI_NET_BUY":     _reindex(ofi_buy,     0.0),
        "OFI_NET_SELL":    _reindex(ofi_sell,    0.0),
        "OB_IMBAL":        _reindex(ob_imbal,    0.0),
        "OB_IMBAL_BUY":    _reindex(ob_buy,      0.5),
        "OB_IMBAL_SELL":   _reindex(ob_sell,     0.5),
        "DP_MICRO":        _reindex(dp_micro,    0.0),
        "CANCEL_NET":      _reindex(cancel_net,  0.0),
        "CANCEL_NET_BUY":  _reindex(c_buy,       0.0),
        "CANCEL_NET_SELL": _reindex(c_sell,      0.0),
        "SPREAD":          _reindex(spread,      np.nan),
        "DEPTH5":          _reindex(depth5,      np.nan),
        "RV_PROXY":        rv_proxy,
        "CANCEL_RATE":     _reindex(cancel_rate, 0.0),
        "LIFE_TIME":       _reindex(life_time,   np.nan),
    }

    if norm_params is None:
        return raw

    return {
        name: zscore(s, norm_params[name]["mean"], norm_params[name]["std"])
        if name in norm_params else s
        for name, s in raw.items()
    }


def segment_mask(index: pd.DatetimeIndex, segment: str) -> np.ndarray:
    """供 point_cloud.py 使用：对 DatetimeIndex 做时段掩码。"""
    idx_sh = index.tz_convert("Asia/Shanghai")
    # 向量化算 (h*3600 + m*60 + s)，比逐元素 Python 循环快 100x+
    local_s = (idx_sh.hour.values * 3600
               + idx_sh.minute.values * 60
               + idx_sh.second.values).astype(np.int64)
    return _trading_mask(local_s, segment)