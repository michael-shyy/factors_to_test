"""cluster_stats.py — 坍缩事件聚集性（CV）。

时间戳规则：基于已识别的历史事件时间序列，无前视。
"""
import numpy as np
import pandas as pd


def collapse_cv(events: pd.DataFrame) -> dict[str, float]:
    """
    计算买/卖侧坍缩事件到达间隔的变异系数。

    Returns
    -------
    dict 含 CV_bid, CV_ask, COLLAPSE_CV_ASYM
    """
    def _cv(times: np.ndarray) -> float:
        if len(times) < 3:
            return np.nan
        dt = np.diff(np.sort(times))
        m = np.mean(dt)
        if m == 0:
            return np.nan
        return float(np.std(dt) / m)

    cv_bid = _cv(events[events['side'] == 'bid']['time'].values) if len(events) > 0 else np.nan
    cv_ask = _cv(events[events['side'] == 'ask']['time'].values) if len(events) > 0 else np.nan

    asym = (float(cv_bid - cv_ask)
            if not (np.isnan(cv_bid) or np.isnan(cv_ask)) else np.nan)

    return {'CV_bid': cv_bid, 'CV_ask': cv_ask, 'COLLAPSE_CV_ASYM': asym}
