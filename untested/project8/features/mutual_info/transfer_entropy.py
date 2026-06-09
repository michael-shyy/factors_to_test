"""transfer_entropy.py — Transfer Entropy（有向信息流）。

TE(X→Y) = MI( Y_future ; X_past | Y_past )
用 KSG 的条件 MI 版本近似：
  TE(X→Y) ≈ MI(Y_t, X_{t-1}) - MI(Y_t, Y_{t-1})

时间戳规则：全部使用滞后变量，无前视。
"""
import numpy as np
from .ksg_estimator import ksg_mi


def transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 1,
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Returns
    -------
    (TE_x2y, TE_y2x, TE_DIR)
    TE_DIR = TE(X→Y) - TE(Y→X)，正值表示 X 领先驱动 Y。
    """
    n = min(len(x), len(y))
    if n < lag + k + 5:
        return np.nan, np.nan, np.nan

    y_curr   = y[lag:]
    x_lagged = x[:-lag]
    y_lagged = y[:-lag]
    length = len(y_curr)

    valid = (np.isfinite(y_curr) & np.isfinite(x_lagged) & np.isfinite(y_lagged))
    if valid.sum() < k + 5:
        return np.nan, np.nan, np.nan

    mi_xy = ksg_mi(y_curr[valid], x_lagged[valid], k)
    mi_yy = ksg_mi(y_curr[valid], y_lagged[valid], k)
    te_x2y = max(mi_xy - mi_yy, 0.0)

    # 反向
    x_curr   = x[lag:]
    y_lag2   = y[:-lag]
    x_lag2   = x[:-lag]
    valid2 = np.isfinite(x_curr) & np.isfinite(y_lag2) & np.isfinite(x_lag2)
    if valid2.sum() >= k + 5:
        mi_yx = ksg_mi(x_curr[valid2], y_lag2[valid2], k)
        mi_xx = ksg_mi(x_curr[valid2], x_lag2[valid2], k)
        te_y2x = max(mi_yx - mi_xx, 0.0)
    else:
        te_y2x = np.nan

    te_dir = (float(te_x2y - te_y2x)
              if not (np.isnan(te_x2y) or np.isnan(te_y2x)) else np.nan)
    return float(te_x2y), float(te_y2x), te_dir
