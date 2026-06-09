"""
dict_stability.py — 小时词典 Jaccard 相似度时序。

时间戳规则：按 trade.TradeTime 分小时切片，每小时只用该小时已发生的数据，无前视。
"""
import numpy as np
from collections import Counter


def _extract_substrings(seq: np.ndarray, min_count: int = 5) -> set:
    """提取长度 2–4 的子串，保留出现次数 > min_count 的子串集合。"""
    result = set()
    n = len(seq)
    for length in (2, 3, 4):
        if n < length:
            continue
        # stride trick: shape (n-length+1, length)
        strides = np.lib.stride_tricks.as_strided(
            seq,
            shape=(n - length + 1, length),
            strides=(seq.strides[0], seq.strides[0]),
        )
        # encode each row as a unique integer key using a base large enough
        base = int(seq.max()) + 1 if len(seq) > 0 else 2
        # use np.unique with counts
        # convert rows to scalar keys
        powers = base ** np.arange(length, dtype=np.int64)
        keys = strides.astype(np.int64) @ powers
        unique_keys, counts = np.unique(keys, return_counts=True)
        frequent_keys = unique_keys[counts > min_count]
        # decode back to tuples
        for key in frequent_keys:
            tup = []
            remaining = int(key)
            for _ in range(length):
                tup.append(remaining % base)
                remaining //= base
            result.add(tuple(tup))
    return result


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def dict_stability_daily(
    seq: np.ndarray,
    timestamps: np.ndarray,
    min_count: int = 5,
) -> tuple[float, float]:
    """
    按小时切片序列，计算相邻小时词典 Jaccard 相似度序列，
    返回 (mean, std)，对应 LZ_dict_stability 和 LZ_dict_volatility。

    Parameters
    ----------
    seq        : 符号序列（整数数组）
    timestamps : 对应的 UNIX 秒时间戳（与 seq 等长）
    min_count  : 子串进入词典的最低频次

    Returns
    -------
    (LZ_dict_stability, LZ_dict_volatility)，数据不足时返回 (nan, nan)。
    """
    if len(seq) < 50:
        return np.nan, np.nan

    # 按小时分组
    hours = ((timestamps + 28800) % 86400) // 3600  # 本地小时
    unique_hours = sorted(set(hours.astype(int)))

    hourly_dicts = []
    for h in unique_hours:
        mask = hours == h
        sub = seq[mask]
        if len(sub) >= 20:
            hourly_dicts.append(_extract_substrings(sub, min_count))

    if len(hourly_dicts) < 2:
        return np.nan, np.nan

    sims = [jaccard(hourly_dicts[i], hourly_dicts[i + 1])
            for i in range(len(hourly_dicts) - 1)]

    return float(np.mean(sims)), float(np.std(sims))
