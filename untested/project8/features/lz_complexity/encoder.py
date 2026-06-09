"""
encoder.py — 将逐笔成交序列编码为 LZ76 所需的符号串。

符号体系
--------
二值编码：买=1, 卖=0
六符号编码：
  A=买大单, B=买中单, C=买小单
  D=卖大单, E=卖中单, F=卖小单
"""
import numpy as np
import pandas as pd

# 六符号映射
_SIX_MAP = {
    (0, 2): 0,  # A: 买大
    (0, 1): 1,  # B: 买中
    (0, 0): 2,  # C: 买小
    (1, 2): 3,  # D: 卖大
    (1, 1): 4,  # E: 卖中
    (1, 0): 5,  # F: 卖小
}


def size_bucket(volume: np.ndarray, p_small: float = 0.2, p_large: float = 0.8) -> np.ndarray:
    """
    按成交量分位数划分大/中/小单：0=小，1=中，2=大。
    仅使用传入序列自身分布（日内动态），无前视。
    """
    q_low  = np.nanquantile(volume, p_small)
    q_high = np.nanquantile(volume, p_large)
    bucket = np.ones(len(volume), dtype=np.int8)  # 中单默认=1
    bucket[volume <= q_low]  = 0
    bucket[volume >= q_high] = 2
    return bucket


def encode_binary(bs_flag: np.ndarray) -> np.ndarray:
    """方向编码为 0/1 序列（主买=1, 主卖=0）。"""
    return (bs_flag == 0).astype(np.int8)


def encode_six(bs_flag: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """方向 × 量级 → 六符号整数序列 (0–5)。"""
    bucket = size_bucket(volume)
    result = np.empty(len(bs_flag), dtype=np.int8)
    for i, (b, s) in enumerate(zip(bs_flag.astype(int), bucket)):
        result[i] = _SIX_MAP[(b, int(s))]
    return result


def encode_six_fast(bs_flag: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """encode_six 的向量化版本（无 Python 循环）。"""
    bucket = size_bucket(volume)
    # buy(0)/sell(1) × 3 + size(0/1/2)
    return (bs_flag.astype(np.int8) * 3 + bucket).astype(np.int8)
