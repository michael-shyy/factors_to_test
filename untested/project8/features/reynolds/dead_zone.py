"""
dead_zone.py — Re_OF 相变死区阈值。

在滚动窗口内，对 Re_OF 绝对值 < θ 的部分输出 0，超过阈值才输出信号。
θ 取日内 Re_OF 绝对值的 P30（固定分位数，不做跨样本优化以防过拟合）。

时间戳规则：θ 用全日 Re_OF 分布计算（日末汇总阶段），输出标量因子时无前视。
"""
import numpy as np
import pandas as pd


def apply_dead_zone(re_series: np.ndarray, pct: float = 0.30) -> np.ndarray:
    """
    对 Re_OF 时序应用死区：绝对值 < θ 置 0，其余保留。
    θ = abs(re_series) 的 pct 分位数。
    """
    valid = re_series[np.isfinite(re_series)]
    if len(valid) == 0:
        return re_series.copy()
    theta = np.quantile(np.abs(valid), pct)
    out = re_series.copy().astype(float)
    out[np.abs(out) < theta] = 0.0
    return out


def re_of_daily(
    mlofi_norm: np.ndarray,
    resil_norm: float,
) -> dict[str, float]:
    """
    日频 Re_OF 因子值（MLOFI 时序均值 × 日内恢复速度标量）。

    Parameters
    ----------
    mlofi_norm : Z-score 标准化后的 MLOFI 时序
    resil_norm : 全天平均恢复速度（已归一化）

    Returns
    -------
    dict 含 Re_OF（加死区）和 Re_OF_raw（不加死区）
    """
    if np.isnan(resil_norm):
        return {'Re_OF': np.nan, 'Re_OF_raw': np.nan}

    re_ts = mlofi_norm.values if hasattr(mlofi_norm, 'values') else mlofi_norm
    re_ts = re_ts.astype(float) * resil_norm

    re_dz = apply_dead_zone(re_ts)
    finite = re_dz[np.isfinite(re_dz)]
    raw    = re_ts[np.isfinite(re_ts)]

    return {
        'Re_OF':     float(np.mean(finite)) if len(finite) > 0 else np.nan,
        'Re_OF_raw': float(np.mean(raw))    if len(raw)    > 0 else np.nan,
    }
