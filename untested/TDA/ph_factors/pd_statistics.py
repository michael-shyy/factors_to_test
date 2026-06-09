"""13 个 PD 统计量，H0/H1 通用。

所有函数输入为 (k, 2) ndarray（birth, death 列），theta_null 为该月标定值。
边界情况：K==0 全返回 NaN；K==1 部分返回 NaN；sum(pers)==0 特殊处理。
"""
from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats
from typing import Dict


def _pers(pd: np.ndarray) -> np.ndarray:
    """提取持久性数组。"""
    if len(pd) == 0:
        return np.empty(0, dtype=np.float64)
    return pd[:, 1] - pd[:, 0]


def compute_pd_stats(pd_arr: np.ndarray, theta_null: float) -> Dict[str, float]:
    """
    计算全部 13 个统计量。

    Args:
        pd_arr:     (k, 2) float64，(birth, death) 对
        theta_null: 该股票该月的 H1 噪声阈值

    Returns:
        dict，key 为统计量名，value 为 float（可能为 NaN）
    """
    nan = float("nan")
    K = len(pd_arr)

    if K == 0:
        return {s: nan for s in [
            "MAXPERS", "TOP3_MEAN", "L2", "MAXPERS_RATIO", "ENT",
            "P75_MINUS_P25", "SKEW", "COUNT_LOW", "COUNT_HIGH", "MAXGAP",
            "MEAN_BIRTH", "MEAN_DEATH", "LIFETIME_OVER_BIRTH_MEAN",
        ]}

    pers  = _pers(pd_arr)
    birth = pd_arr[:, 0]
    death = pd_arr[:, 1]
    s_pers = np.sort(pers)[::-1]  # 降序
    total  = float(np.sum(pers))

    # 1. MAXPERS
    maxpers = float(np.max(pers))

    # 2. TOP3_MEAN
    top3_mean = float(np.mean(s_pers[:3]))

    # 3. L2
    l2 = float(np.sum(pers ** 2))

    # 4. MAXPERS_RATIO
    maxpers_ratio = (maxpers / total) if total > 0 else nan

    # 5. ENT
    if total > 0:
        p = pers / total
        ent = float(-np.sum(p * np.log(np.maximum(p, 1e-12))))
    else:
        ent = 0.0

    # 6. P75_MINUS_P25
    p75_p25 = float(np.percentile(pers, 75) - np.percentile(pers, 25)) if K >= 2 else nan

    # 7. SKEW —— 持久性几乎相同时 scipy 三阶矩会精度溢出，直接返回 NaN
    if K >= 2 and pers.std() > 1e-10 * max(abs(pers.mean()), 1e-10):
        skew = float(sp_stats.skew(pers))
    else:
        skew = nan

    # 8. COUNT_LOW
    count_low = int((pers > 1.0 * theta_null).sum())

    # 9. COUNT_HIGH
    count_high = int((pers > 3.0 * theta_null).sum())

    # 10. MAXGAP
    # s_pers 为降序，相邻差 diff 为负；最大间隙 = 最大的相邻落差 = max(-diff)
    if K >= 2:
        maxgap = float(np.max(-np.diff(s_pers)))
    else:
        maxgap = nan

    # 11. MEAN_BIRTH
    mean_birth = float(np.mean(birth))

    # 12. MEAN_DEATH
    mean_death = float(np.mean(death))

    # 13. LIFETIME_OVER_BIRTH_MEAN
    lifetime_over_birth = float(np.mean(pers / (birth + 1e-9)))

    return {
        "MAXPERS":                  maxpers,
        "TOP3_MEAN":                top3_mean,
        "L2":                       l2,
        "MAXPERS_RATIO":            maxpers_ratio,
        "ENT":                      ent,
        "P75_MINUS_P25":            p75_p25,
        "SKEW":                     skew,
        "COUNT_LOW":                float(count_low),
        "COUNT_HIGH":               float(count_high),
        "MAXGAP":                   maxgap,
        "MEAN_BIRTH":               mean_birth,
        "MEAN_DEATH":               mean_death,
        "LIFETIME_OVER_BIRTH_MEAN": lifetime_over_birth,
    }
