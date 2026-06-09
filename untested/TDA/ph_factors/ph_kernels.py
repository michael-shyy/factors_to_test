"""ripser 调用封装 + PD 后处理。

注意：
- ripser 输入必须是 np.float64
- distance_matrix=True 时矩阵必须严格对称，对角线为 0
- H0 输出包含一个 (0, inf) 全局连通团，需过滤
"""
from __future__ import annotations
import numpy as np
from typing import Tuple
from ripser import ripser


def compute_ph(
    dist_matrix: np.ndarray,
    maxdim: int = 1,
    thresh: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对距离矩阵计算持续同调。

    Args:
        dist_matrix: (M, M) float64 对称距离矩阵
        maxdim:      最高同调维数，默认 1
        thresh:      ripser 距离截断,只算 birth < thresh 的同调结构。
                     用于点云高度退化(大量近重合点)的配置,可大幅压缩 ripser 耗时。
                     None 表示不截断(=inf)。

    Returns:
        pd_h0: (k0, 2) H0 (birth, death) 对，已过滤 (0, inf) 全局团
        pd_h1: (k1, 2) H1 (birth, death) 对
    """
    dist_matrix = np.asarray(dist_matrix, dtype=np.float64)
    # 保证严格对称
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0.0)

    rip_kwargs = {"distance_matrix": True, "maxdim": maxdim}
    if thresh is not None and np.isfinite(thresh) and thresh > 0:
        rip_kwargs["thresh"] = float(thresh)
    result = ripser(dist_matrix, **rip_kwargs)
    dgms = result["dgms"]

    # H0：过滤掉 death == inf 的全局连通团
    h0 = dgms[0]
    pd_h0 = h0[np.isfinite(h0[:, 1])] if len(h0) > 0 else np.empty((0, 2), dtype=np.float64)

    # H1
    pd_h1 = dgms[1] if maxdim >= 1 and len(dgms) > 1 else np.empty((0, 2), dtype=np.float64)
    # 过滤 inf（理论上 H1 不应有 inf，但防御性处理）
    if len(pd_h1) > 0:
        pd_h1 = pd_h1[np.isfinite(pd_h1).all(axis=1)]

    return (
        np.asarray(pd_h0, dtype=np.float64),
        np.asarray(pd_h1, dtype=np.float64),
    )