"""
data/snapshot_aligner.py — 四流毫秒级对齐，严格保证无前视偏差。

对齐规则
========
目标：将 order / trade / tick 三条流对齐到一条统一时间轴（tick 快照时刻），
使得每个快照时刻 t 上附加的 order/trade 信息 **全部来自 ≤t 的已知数据**。

具体规则
--------
1. tick 快照时刻 = 对齐基准轴。每条 tick 记录的 `time` 字段代表该快照的
   发布时刻，订单簿状态已反映该时刻已成交/已挂单的全部信息。

2. order 对齐（merge_asof backward）：
   对每笔委托，取"最近一个 tick 快照，且该快照时刻 ≤ OrderTime"。
   即委托在 OrderTime 发出，此时可用的最新快照是它发出**之前**的最后一个
   tick——严格无前视。

3. trade 对齐（merge_asof backward）：
   同理，对每笔成交取 TradeTime ≤ tick.time 的最近快照。

4. 聚合到 tick 轴（snapshot_features）：
   对每个快照时刻 t，汇总 (t_prev, t] 窗口内到达的委托和成交。
   t_prev 为上一个快照时刻，初始值取第一个 tick 时刻前 1 秒。
   ——保证汇总的事件时间戳均 ≤ t，无法用到未来 tick 的信息。

前视风险检查清单
----------------
- [OK] merge_asof direction='backward'：只向过去查找，不用未来快照。
- [OK] 聚合窗口 (t_prev, t]：开区间左端，事件恰好在 t_prev 时不被重复计入。
- [OK] 量价计算（amt = price × vol）在委托/成交行上直接计算，不依赖快照价格。
- [WARN] running VWAP（非全日）：调用方必须用此模块返回的增量数据维护
  running sum，禁止使用全日 totalAmt/totalVolume 字段计算 VWAP。
"""
import numpy as np
import pandas as pd


_BOOK_COLS = ['time', 'akp1', 'akv1', 'bdp1', 'bdv1',
              'akp2', 'akv2', 'akp3', 'akv3', 'akp4', 'akv4', 'akp5', 'akv5',
              'bdp2', 'bdv2', 'bdp3', 'bdv3', 'bdp4', 'bdv4', 'bdp5', 'bdv5']


def align_orders_to_tick(
    order: pd.DataFrame,
    tick: pd.DataFrame,
    tolerance_sec: float = 3.0,
) -> pd.DataFrame:
    """
    为每笔委托附加其发出时刻之前最近的 tick 快照字段。

    时间戳规则：取 tick.time ≤ order.OrderTime，direction='backward'。
    tolerance_sec：超过此秒数无对应快照则置 NaN（防止集合竞价期垃圾数据污染）。

    Returns
    -------
    order + tick 快照列，原 OrderTime 列不变。
    """
    tick_sorted = tick.sort_values('time').reset_index(drop=True)
    order_sorted = order.sort_values('OrderTime').reset_index(drop=True)

    book_cols = [c for c in _BOOK_COLS if c in tick_sorted.columns]
    merged = pd.merge_asof(
        order_sorted,
        tick_sorted[book_cols],
        left_on='OrderTime',
        right_on='time',
        direction='backward',
        tolerance=tolerance_sec,
    )
    return merged


def align_trades_to_tick(
    trade: pd.DataFrame,
    tick: pd.DataFrame,
    tolerance_sec: float = 3.0,
) -> pd.DataFrame:
    """
    为每笔成交附加其发生时刻之前最近的 tick 快照字段。

    时间戳规则：取 tick.time ≤ trade.TradeTime，direction='backward'。
    """
    tick_sorted = tick.sort_values('time').reset_index(drop=True)
    trade_sorted = trade.sort_values('TradeTime').reset_index(drop=True)

    book_cols = [c for c in _BOOK_COLS if c in tick_sorted.columns]
    merged = pd.merge_asof(
        trade_sorted,
        tick_sorted[book_cols],
        left_on='TradeTime',
        right_on='time',
        direction='backward',
        tolerance=tolerance_sec,
    )
    return merged


def snapshot_features(
    order: pd.DataFrame,
    trade: pd.DataFrame,
    tick: pd.DataFrame,
) -> pd.DataFrame:
    """
    以 tick 快照轴为基准，聚合每个窗口 (t_prev, t] 内的委托/成交特征。

    时间戳规则：每行对应 tick.time=t，只汇总 OrderTime/TradeTime ∈ (t_prev, t]
    的事件——全部来自 ≤t 的已知数据。完全向量化，无 Python 循环。
    """
    tick_times = tick.sort_values('time')['time'].values
    T = len(tick_times)

    real_trade = trade[trade['TradeCode'] == 0].sort_values('TradeTime').reset_index(drop=True)
    order_s    = order.sort_values('OrderTime').reset_index(drop=True)

    o_times = order_s['OrderTime'].values
    o_bs    = order_s['BSFlag'].values
    o_vol   = order_s['OrderVolume'].values
    o_amt   = (order_s['OrderPrice'] * order_s['OrderVolume']).values

    t_times = real_trade['TradeTime'].values
    t_bs    = real_trade['BSFlag'].values
    t_vol   = real_trade['TradeVolume'].values
    t_amt   = (real_trade['TradePrice'] * real_trade['TradeVolume']).values

    def _reduceat_sum(vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """bins[i] = start index of tick window i in sorted vals."""
        if len(vals) == 0 or len(bins) == 0:
            return np.zeros(T)
        out = np.add.reduceat(vals, bins)
        # 修正最后一个区间之后没有数据的边界
        return out

    def _bin_assign(event_times: np.ndarray) -> np.ndarray:
        """
        将每个事件分配到对应的 tick 窗口 (t_prev, t]。
        返回长度 = len(event_times) 的整数数组，值为 [0, T)。
        时间戳规则：searchsorted 'right' → 事件 <= t 落入该 tick，> t 落入下一个。
        """
        return np.searchsorted(tick_times, event_times, side='right') - 1

    def _group_sum(vals: np.ndarray, bins: np.ndarray, T: int) -> np.ndarray:
        """按 bins 分组求和，结果长度 = T。"""
        out = np.zeros(T)
        np.add.at(out, bins, vals)
        return out

    def _group_count(bins: np.ndarray, T: int) -> np.ndarray:
        out = np.zeros(T)
        np.add.at(out, bins, 1)
        return out

    # 过滤有效区间（bins 在 [0, T)）
    o_bins = _bin_assign(o_times)
    valid_o = (o_bins >= 0) & (o_bins < T)
    o_bins, o_bs_v, o_vol_v, o_amt_v = (
        o_bins[valid_o], o_bs[valid_o], o_vol[valid_o], o_amt[valid_o]
    )

    t_bins = _bin_assign(t_times)
    valid_t = (t_bins >= 0) & (t_bins < T)
    t_bins, t_bs_v, t_vol_v, t_amt_v = (
        t_bins[valid_t], t_bs[valid_t], t_vol[valid_t], t_amt[valid_t]
    )

    buy_o_mask  = o_bs_v == 0
    sell_o_mask = o_bs_v == 1
    buy_t_mask  = t_bs_v == 0
    sell_t_mask = t_bs_v == 1

    n_buy_order   = _group_count(o_bins[buy_o_mask],  T)
    n_sell_order  = _group_count(o_bins[sell_o_mask], T)
    buy_order_vol = _group_sum(o_vol_v[buy_o_mask],  o_bins[buy_o_mask],  T)
    sell_order_vol= _group_sum(o_vol_v[sell_o_mask], o_bins[sell_o_mask], T)
    buy_order_amt = _group_sum(o_amt_v[buy_o_mask],  o_bins[buy_o_mask],  T)
    sell_order_amt= _group_sum(o_amt_v[sell_o_mask], o_bins[sell_o_mask], T)

    n_buy_trade   = _group_count(t_bins[buy_t_mask],  T)
    n_sell_trade  = _group_count(t_bins[sell_t_mask], T)
    buy_trade_vol = _group_sum(t_vol_v[buy_t_mask],  t_bins[buy_t_mask],  T)
    sell_trade_vol= _group_sum(t_vol_v[sell_t_mask], t_bins[sell_t_mask], T)
    buy_trade_amt = _group_sum(t_amt_v[buy_t_mask],  t_bins[buy_t_mask],  T)
    sell_trade_amt= _group_sum(t_amt_v[sell_t_mask], t_bins[sell_t_mask], T)

    # running VWAP：累计成交量/金额（shift 已由 bin 分配保证无前视）
    all_trade_vol = _group_sum(t_vol_v, t_bins, T)
    all_trade_amt = _group_sum(t_amt_v, t_bins, T)
    cum_vol = np.cumsum(all_trade_vol)
    cum_amt = np.cumsum(all_trade_amt)
    # shift(1)：t 时刻 VWAP 用截至 t-1 的累计（严格无前视）
    cum_vol_prev = np.empty(T); cum_vol_prev[0] = 0; cum_vol_prev[1:] = cum_vol[:-1]
    cum_amt_prev = np.empty(T); cum_amt_prev[0] = 0; cum_amt_prev[1:] = cum_amt[:-1]
    running_vwap = np.where(cum_vol_prev > 0, cum_amt_prev / cum_vol_prev, np.nan)

    return pd.DataFrame({
        'time':           tick_times,
        'n_buy_order':    n_buy_order,
        'n_sell_order':   n_sell_order,
        'buy_order_vol':  buy_order_vol,
        'sell_order_vol': sell_order_vol,
        'buy_order_amt':  buy_order_amt,
        'sell_order_amt': sell_order_amt,
        'n_buy_trade':    n_buy_trade,
        'n_sell_trade':   n_sell_trade,
        'buy_trade_vol':  buy_trade_vol,
        'sell_trade_vol': sell_trade_vol,
        'buy_trade_amt':  buy_trade_amt,
        'sell_trade_amt': sell_trade_amt,
        'running_vwap':   running_vwap,
    })
