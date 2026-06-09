"""
regime_classifier.py — 高/低波动状态分类。

时间戳规则：rolling std 使用 shift(1) 滚动，t 时刻的 vol 标签只用 ≤t-1 的数据。
"""
import numpy as np
import pandas as pd


def rolling_vol_label(
    prices: np.ndarray,
    timestamps: np.ndarray,
    window_sec: int = 300,
    tick_interval: float = 4.0,
) -> np.ndarray:
    """
    滚动 window_sec 秒的成交价标准差，中位数为切分阈值。
    返回每笔的 vol 状态标签：1=高波动，0=低波动，-1=未知（窗口不足）。

    时间戳规则：用 shift(1) 确保 t 时刻不用自身数据。
    """
    n = len(prices)
    window_ticks = max(int(window_sec / tick_interval), 5)
    s = pd.Series(prices)
    roll_std = s.shift(1).rolling(window_ticks, min_periods=5).std().values

    # 全天中位数为阈值
    median_vol = np.nanmedian(roll_std)

    labels = np.full(n, -1, dtype=np.int8)
    valid = ~np.isnan(roll_std)
    labels[valid & (roll_std >= median_vol)] = 1
    labels[valid & (roll_std <  median_vol)] = 0
    return labels


def split_by_regime(
    seq: np.ndarray,
    vol_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (high_vol_seq, low_vol_seq)，按 vol_labels 过滤。"""
    return seq[vol_labels == 1], seq[vol_labels == 0]
