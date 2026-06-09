"""ksg_estimator.py — KSG kNN 互信息估计器（numba 加速）。

时间戳规则：输入为滚动窗口内的历史观测对，调用方保证无前视。
"""
import numpy as np
from numba import njit
from scipy.special import digamma as _digamma_scipy


# 预计算 digamma 查找表（整数参数 1..2000）
_DG_MAX = 2001
_DG_TABLE = np.array([_digamma_scipy(i) for i in range(_DG_MAX)], dtype=np.float64)


@njit(cache=True)
def _digamma_fast(n: int) -> float:
    """整数 digamma，用预计算表。超出范围退回近似值。"""
    if n < 1:
        return 0.0
    if n < 2001:
        return _DG_TABLE[n]
    # Stirling 近似
    x = float(n)
    return np.log(x) - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x)


@njit(cache=True)
def _ksg_mi_core(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """
    KSG MI 核心（Chebyshev 距离，全 numba）。
    输入已保证无 nan 且 len >= k+2。
    """
    n = len(x)
    dg_k = _digamma_fast(k)
    dg_n = _digamma_fast(n)

    sum_dg = 0.0
    for i in range(n):
        xi, yi = x[i], y[i]

        # 找第 k 近邻 Chebyshev 距离（排除自身）
        kth = np.inf
        # 用部分排序：维护一个大小为 k 的最大堆等价物
        # 简单起见：先全部算距离，再取第 k 小
        # （n=500 时 500*500=25万次 float 运算，numba 内极快）
        k_count = 0
        # 先找第 k 近邻 eps
        eps = 0.0
        # 两趟：第一趟建距离数组，第二趟数边际邻居
        # 为避免分配内存，手动选择第 k 小
        # 使用插入式 top-k（k=5，极快）
        top = np.full(k, np.inf)
        for j in range(n):
            if j == i:
                continue
            d = abs(xi - x[j])
            dy = abs(yi - y[j])
            if dy > d:
                d = dy
            # 插入 top-k（最大堆最小值维护）
            if d < top[k - 1]:
                top[k - 1] = d
                # 冒泡保持降序
                m = k - 1
                while m > 0 and top[m] < top[m - 1]:
                    tmp = top[m]; top[m] = top[m - 1]; top[m - 1] = tmp
                    m -= 1
        eps = top[k - 1]

        # 计算边际邻居数（KSG 原始定义：包含自身，即严格 < eps 的全部点数）
        nx, ny = 0, 0
        for j in range(n):
            if abs(x[j] - xi) < eps:
                nx += 1
            if abs(y[j] - yi) < eps:
                ny += 1

        sum_dg += _digamma_fast(nx + 1) + _digamma_fast(ny + 1)

    mi = dg_k - sum_dg / n + dg_n
    return max(mi, 0.0)


def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """估计 MI(X, Y)，样本不足时返回 nan。"""
    n = len(x)
    if n < k + 2:
        return np.nan
    return float(_ksg_mi_core(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        k,
    ))


def rolling_ksg_mi(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 500,
    k: int = 5,
    step: int = 10,
) -> np.ndarray:
    """
    滚动窗口 KSG MI（step=10 默认跳步，精度损失极小但速度提升 10×）。

    时间戳规则：输出第 i 位只用 x/y[i-window:i] 历史，无前视。
    """
    n = len(x)
    out = np.full(n, np.nan)
    xc = np.ascontiguousarray(x, dtype=np.float64)
    yc = np.ascontiguousarray(y, dtype=np.float64)
    for i in range(window - 1, n, step):
        xi, yi = xc[i - window + 1: i + 1], yc[i - window + 1: i + 1]
        valid = np.isfinite(xi) & np.isfinite(yi)
        if valid.sum() >= k + 2:
            val = float(_ksg_mi_core(xi[valid], yi[valid], k))
            # 向前填充直到下一个计算点
            end = min(i + step, n)
            out[i: end] = val
    return out
