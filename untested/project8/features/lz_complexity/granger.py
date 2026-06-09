"""
granger.py — 买卖 LZ 变化量的 Granger 因果检验。

时间戳规则：输入为历史 LZ 时序（已计算完毕的 rolling 序列），无原始数据前视。
"""
import numpy as np


def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    """OLS 残差平方和，用 numpy lstsq 避免 LAPACK 警告。"""
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(resid @ resid)


def _granger_f(y: np.ndarray, x: np.ndarray, lags: int = 2) -> float:
    n = len(y)
    if n < lags * 3 + 5:
        return np.nan

    Y = y[lags:]
    m = len(Y)
    ones = np.ones((m, 1))
    Y_lags = np.column_stack([y[lags - l - 1:n - l - 1] for l in range(lags)])
    X_lags = np.column_stack([x[lags - l - 1:n - l - 1] for l in range(lags)])

    try:
        rss_r = _ols_rss(np.hstack([ones, Y_lags]), Y)
        rss_u = _ols_rss(np.hstack([ones, Y_lags, X_lags]), Y)
        if rss_u <= 0:
            return np.nan
        f = ((rss_r - rss_u) / lags) / (rss_u / (m - 2 * lags - 1))
        return float(f)
    except Exception:
        return np.nan


def granger_direction(
    lz_buy: np.ndarray,
    lz_sell: np.ndarray,
    max_lag: int = 3,
    f_threshold: float = 3.84,
) -> int:
    """
    +1 = 买方领先，-1 = 卖方领先，0 = 无显著因果。
    """
    valid = ~(np.isnan(lz_buy) | np.isnan(lz_sell))
    b = np.diff(lz_buy[valid])
    s = np.diff(lz_sell[valid])
    if len(b) < 20:
        return 0

    f_bs = max(_granger_f(s, b, lags=lag) or 0.0 for lag in range(1, max_lag + 1))
    f_sb = max(_granger_f(b, s, lags=lag) or 0.0 for lag in range(1, max_lag + 1))

    if f_bs > f_threshold and f_bs > f_sb:
        return 1
    if f_sb > f_threshold and f_sb > f_bs:
        return -1
    return 0
