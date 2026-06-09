"""
transition_matrix.py — 一阶转移矩阵与 KL 散度。

时间戳规则：输入为滚动窗口内的历史符号序列，调用方保证无前视。
"""
import numpy as np

_EPS = 0.01  # Laplace 平滑


def build_transition_matrix(seq: np.ndarray, k: int = 6) -> np.ndarray:
    """
    统计 k × k 一阶转移计数矩阵，并做 Laplace 平滑（+eps）后归一化为概率。
    """
    counts = np.full((k, k), _EPS)
    for i in range(len(seq) - 1):
        counts[seq[i], seq[i + 1]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def kl_divergence_uniform(trans: np.ndarray) -> float:
    """
    计算转移矩阵与均匀矩阵（每行等概率 1/k）的 KL 散度之和。
    KL(P || Q) = sum P log(P/Q)，Q = 1/k。
    """
    k = trans.shape[0]
    q = 1.0 / k
    return float(np.sum(trans * np.log(trans / q)))


def transition_kl_asym(seq: np.ndarray, k: int = 6) -> tuple[float, float, float]:
    """
    计算全序列、买方子序列、卖方子序列的 KL 散度，返回 (kl_total, kl_buy, kl_sell)。

    六符号编码中：0,1,2 = 买（A/B/C），3,4,5 = 卖（D/E/F）。
    买方子矩阵取前 3×3，卖方取后 3×3（各自归一化）。
    """
    if len(seq) < 20:
        return np.nan, np.nan, np.nan

    trans = build_transition_matrix(seq, k)
    kl_total = kl_divergence_uniform(trans)

    # 买方子矩阵（状态 0,1,2 之间的转移）
    buy_mask = seq < 3
    buy_seq  = seq[buy_mask]
    if len(buy_seq) > 10:
        buy_trans = build_transition_matrix(buy_seq, k=3)
        kl_buy = kl_divergence_uniform(buy_trans)
    else:
        kl_buy = np.nan

    # 卖方子矩阵（状态 3,4,5 → 映射为 0,1,2）
    sell_mask = seq >= 3
    sell_seq  = (seq[sell_mask] - 3).astype(np.int8)
    if len(sell_seq) > 10:
        sell_trans = build_transition_matrix(sell_seq, k=3)
        kl_sell = kl_divergence_uniform(sell_trans)
    else:
        kl_sell = np.nan

    return kl_total, kl_buy, kl_sell
