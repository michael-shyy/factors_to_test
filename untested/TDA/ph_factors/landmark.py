"""Maxmin landmark 子采样（numba JIT）。

选取 M 个代表点，返回 landmarks、索引、距离矩阵。
随机种子由调用方传入，保证复现性。
"""
from __future__ import annotations
import numpy as np
import numba as nb
from typing import Tuple


@nb.njit(cache=True)
def _maxmin_indices(points: np.ndarray, M: int, first_idx: int) -> np.ndarray:
    """纯 numba 实现 maxmin 选点，返回 M 个 landmark 的索引。

    maxmin 只需比较距离大小，不需要真实距离值，故全程用平方距离，
    省去 N*M 次开方（选点结果完全等价）。
    """
    N = points.shape[0]
    D = points.shape[1]
    indices = np.empty(M, dtype=np.int64)
    indices[0] = first_idx

    # min_dist2[i] = 当前已选 landmark 集合到点 i 的最小【平方】距离
    min_dist2 = np.full(N, np.inf)

    for m in range(1, M):
        prev = indices[m - 1]
        best = 0
        best_d = -1.0
        for i in range(N):
            d = 0.0
            for k in range(D):
                diff = points[i, k] - points[prev, k]
                d += diff * diff
            if d < min_dist2[i]:
                min_dist2[i] = d
            # 选点与更新合并到同一遍循环
            if min_dist2[i] > best_d:
                best_d = min_dist2[i]
                best = i
        indices[m] = best

    return indices


@nb.njit(cache=True)
def _pairwise_dist(points: np.ndarray) -> np.ndarray:
    """计算 (M, M) 欧氏距离矩阵。"""
    M = points.shape[0]
    D = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            d = 0.0
            for k in range(points.shape[1]):
                diff = points[i, k] - points[j, k]
                d += diff * diff
            d = d ** 0.5
            D[i, j] = d
            D[j, i] = d
    return D


def maxmin_landmark(
    cloud: np.ndarray,
    M: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从点云中选取 M 个 maxmin landmark 点。

    Args:
        cloud:  (N, d) float64 点云
        M:      landmark 数量，若 N <= M 则直接返回全部点
        seed:   随机种子，用于确定第一个 landmark

    Returns:
        landmarks:        (M, d) landmark 坐标
        landmark_indices: (M,)   在原点云中的索引
        dist_matrix:      (M, M) landmark 间欧氏距离矩阵
    """
    cloud = np.asarray(cloud, dtype=np.float64)
    N = cloud.shape[0]
    if N == 0:
        empty = np.empty((0, cloud.shape[1]), dtype=np.float64)
        return empty, np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float64)

    M = min(M, N)
    rng = np.random.default_rng(seed)
    first_idx = int(rng.integers(0, N))

    indices = _maxmin_indices(cloud, M, first_idx)
    landmarks = cloud[indices]
    dist_matrix = _pairwise_dist(landmarks)
    return landmarks, indices, dist_matrix
