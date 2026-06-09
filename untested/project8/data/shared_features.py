"""
data/shared_features.py — 多个因子族共用的前置特征。

所有函数的时间戳规则
--------------------
输入  : 已经过 snapshot_aligner 处理的 tick 快照 DataFrame（index 无要求）
输出  : 新列附加到 tick 上，每行特征的时间归属 = 该行的 tick.time
严格  : 只使用 ≤ tick.time 的历史信息，窗口均为回望窗口（rolling backward）
"""
import numpy as np
import pandas as pd


def price_probe_flag(
    tick: pd.DataFrame,
    window: int = 5,
    threshold: float = 0.0003,
) -> pd.Series:
    """
    探价前置标记（PROBE_FLAG）。

    定义：若当前 tick 的 mid-price 相对过去 window 个 tick 的 mid-price 变化
    超过 threshold（默认 3bp），则标记为 1，表示可能存在探价行为。

    时间戳规则：rolling(window) 只使用 [i-window, i-1] 历史，i 时刻本身
    不被纳入滚动窗口（shift(1) 取前一期均值）。
    """
    mid = (tick['akp1'] + tick['bdp1']) / 2.0
    past_mid = mid.shift(1).rolling(window, min_periods=1).mean()
    ret = (mid - past_mid) / past_mid.replace(0, np.nan)
    return (ret.abs() > threshold).astype(np.int8).rename('probe_flag')


def tick_stability(
    tick: pd.DataFrame,
    window: int = 10,
) -> pd.Series:
    """
    等待节拍稳定性（TICK_STABILITY）。

    定义：过去 window 个 tick 时间间隔的变异系数（CV = std/mean）的倒数，
    值越大表示节拍越稳定（均匀到达）。

    时间戳规则：只用 [i-window, i-1] 的时间差序列，无前视。
    """
    dt = tick['time'].diff()  # tick 间隔（秒）
    roll_std  = dt.shift(1).rolling(window, min_periods=3).std()
    roll_mean = dt.shift(1).rolling(window, min_periods=3).mean()
    cv = roll_std / roll_mean.replace(0, np.nan)
    return (1.0 / (cv + 1e-8)).rename('tick_stability')


def bob_symm(tick: pd.DataFrame) -> pd.Series:
    """
    盘口左右对称性（BOB_SYMM, Balance Of Book Symmetry）。

    定义：(bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5)
    其中 depth = 前 5 档累计委托量。
    值域 [-1, 1]，正值表示买方更厚，负值表示卖方更厚。

    时间戳规则：直接使用当前快照已公开的委托量，无前视（tick 发布时
    盘口状态已确定）。
    """
    bid_depth = sum(tick.get(f'bdv{i}', 0) for i in range(1, 6))
    ask_depth = sum(tick.get(f'akv{i}', 0) for i in range(1, 6))
    total = bid_depth + ask_depth
    symm = (bid_depth - ask_depth) / total.replace(0, np.nan)
    return symm.rename('bob_symm')


def add_shared_features(tick: pd.DataFrame) -> pd.DataFrame:
    """一次性计算并附加所有共用特征，返回新 DataFrame。"""
    df = tick.copy()
    df['probe_flag']    = price_probe_flag(df).values
    df['tick_stability'] = tick_stability(df).values
    df['bob_symm']      = bob_symm(df).values
    return df
