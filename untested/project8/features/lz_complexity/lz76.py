"""
lz76.py — LZ76 复杂度核心算法。

归一化公式
----------
二值 (alphabet_size=2):  C_norm = C * log2(n) / n
多符号 (alphabet_size=k): C_norm = C * log2(k) * log2(n) / n
"""
import numpy as np


def _lz76_count(seq: np.ndarray) -> int:
    """LZ76 扫描，统计新子串个数 C。O(n²)：用 bytes.find 做子串搜索。

    搜索范围：在 history = seq[0:i] 中找 seq[i:i+k]，严格无前视。
    """
    n = len(seq)
    if n == 0:
        return 0
    s = bytes(seq.tolist())
    c, i, k = 1, 0, 1
    while i + k <= n:
        if s[:i].find(s[i:i + k]) != -1:
            k += 1
        else:
            c += 1
            i += k
            k = 1
    return c


def _rolling_lz76_njit(seq: np.ndarray, window: int, log2_alpha: float, step: int) -> np.ndarray:
    """滚动 LZ76。out[i] 对应 seq[i-window+1:i+1]，前 window-1 个位置填 -1。"""
    n = len(seq)
    out = np.full(n, -1.0)
    log2n = np.log2(window)
    i = window - 1
    while i < n:
        c = _lz76_count(seq[i - window + 1: i + 1])
        out[i] = c * log2n / window if log2_alpha == 1.0 else c * log2_alpha * log2n / window
        i += step
    return out


def lz76_normalized(seq: np.ndarray, alphabet_size: int = 2) -> float:
    """
    计算归一化 LZ76 复杂度。

    时间戳规则：输入 seq 为已截断到目标窗口的历史序列，调用方保证无前视。
    """
    n = len(seq)
    if n < 10:
        return np.nan
    c = _lz76_count(seq)
    if alphabet_size == 2:
        return c * np.log2(max(n, 2)) / n
    return c * np.log2(alphabet_size) * np.log2(max(n, 2)) / n


def rolling_lz76(
    seq: np.ndarray,
    window: int = 500,
    alphabet_size: int = 2,
    step: int = 1,
) -> np.ndarray:
    """
    滚动窗口 LZ76（numba 加速）。

    时间戳规则：输出第 i 个值对应 seq[i-window:i] 这段历史，严格无前视。
    Returns 长度等于 len(seq) 的数组，前 window-1 个位置为 nan。
    """
    seq_i8 = np.ascontiguousarray(seq, dtype=np.int8)
    log2_alpha = float(np.log2(alphabet_size))
    raw = _rolling_lz76_njit(seq_i8, window, log2_alpha, step)
    out = raw.copy()
    out[out == -1.0] = np.nan
    return out


def lz76_subseries(
    seq: np.ndarray,
    mask: np.ndarray,
    window: int = 500,
    alphabet_size: int = 2,
) -> float:
    """对 seq 按 mask 过滤后的子序列计算单次 LZ76。"""
    sub = seq[mask]
    if len(sub) < window // 2:
        return np.nan
    return lz76_normalized(sub[-window:], alphabet_size)
