"""
point_cloud.py
==============
从 signals.py 输出的 Dict[str, pd.Series] 构造点云。
范式 B：纯延迟嵌入；范式 C：延迟嵌入 + 环境维拼接。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from ph_factors.registry import CloudConfig
from ph_factors.signals import segment_mask


def _delay_embed(arr: np.ndarray, tau: int, order: int = 2) -> np.ndarray:
    """延迟嵌入，stride_tricks 零拷贝。返回 (N-(order-1)*tau, order) float64。"""
    N = len(arr)
    min_len = (order - 1) * tau + 1
    if N < min_len:
        return np.empty((0, order), dtype=np.float64)
    rows = N - (order - 1) * tau
    out = np.empty((rows, order), dtype=np.float64)
    for k in range(order):
        start = (order - 1 - k) * tau
        out[:, k] = arr[start: start + rows]
    return out


def build_point_cloud(
    config: CloudConfig,
    segment: str,
    direction: Optional[str],
    signals: Dict[str, pd.Series],
    tau: int,
) -> np.ndarray:
    """
    构造单个点云。

    Args:
        config:    CloudConfig
        segment:   "FULL" | "OPEN"
        direction: None | "BUY" | "SELL"
        signals:   compute_all_signals 的输出（已标准化）
        tau:       该配置该月的 τ（tick 数，即 3 秒桶数）

    Returns:
        (N, d) float64，已过滤 NaN 行
    """
    # 选主信号
    if direction == "BUY":
        key = config.main_signal + "_BUY"
    elif direction == "SELL":
        key = config.main_signal + "_SELL"
    else:
        key = config.main_signal

    main_s = signals.get(key)
    if main_s is None:
        raise KeyError(f"Signal not found: {key}")

    # 时段过滤
    mask = segment_mask(main_s.index, segment)
    main_arr = main_s.values[mask].astype(np.float64)

    embedded = _delay_embed(main_arr, tau, config.embed_order)  # (N', order)
    n_rows = len(embedded)
    if n_rows == 0:
        return np.empty((0, config.total_dim), dtype=np.float64)

    if config.paradigm == "B" or not config.env_dims:
        cloud = embedded
    else:
        # 范式 C：拼接环境维（取与嵌入后对齐的尾部）
        env_cols = []
        for env_name in config.env_dims:
            env_s = signals.get(env_name)
            if env_s is None:
                raise KeyError(f"Env signal not found: {env_name}")
            env_arr = env_s.values[mask].astype(np.float64)
            offset = len(env_arr) - n_rows
            if offset < 0:
                return np.empty((0, config.total_dim), dtype=np.float64)
            env_cols.append(env_arr[offset:].reshape(-1, 1))
        cloud = np.hstack([embedded] + env_cols)

    # 过滤含 NaN 的行
    valid = ~np.isnan(cloud).any(axis=1)
    return cloud[valid].astype(np.float64)
