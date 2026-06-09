"""size_sequence.py — 爆发内部量级序列结构（批量向量化）。"""
import numpy as np
from .segmenter import Segment


def burst_type_ratios(
    bursts: list[Segment],
    volumes_all: np.ndarray,
) -> dict[str, float]:
    if not bursts:
        return {'prog_ratio': np.nan, 'shock_ratio': np.nan, 'decay_ratio': np.nan}

    q_large = np.nanquantile(volumes_all, 0.8)
    q_small = np.nanquantile(volumes_all, 0.2)

    prog = shock = decay = 0
    for seg in bursts:
        vols = np.asarray(seg.vol_seq, dtype=float)
        if len(vols) < 2:
            continue
        labels = np.where(vols >= q_large, 2, np.where(vols <= q_small, 0, 1))
        if labels[0] >= 2:
            shock += 1
        elif np.all(np.diff(labels.astype(np.int8)) >= 0):
            prog += 1
        elif np.all(np.diff(labels.astype(np.int8)) <= 0):
            decay += 1

    total = len(bursts)
    return {
        'prog_ratio':  prog  / total,
        'shock_ratio': shock / total,
        'decay_ratio': decay / total,
    }


def classify_burst(seg: Segment, q_large: float, q_small: float) -> str:
    """保留接口兼容，内部复用 burst_type_ratios 逻辑。"""
    vols = np.asarray(seg.vol_seq, dtype=float)
    if len(vols) < 2:
        return 'mixed'
    labels = np.where(vols >= q_large, 2, np.where(vols <= q_small, 0, 1))
    if labels[0] >= 2:
        return 'shock'
    diffs = np.diff(labels.astype(np.int8))
    if np.all(diffs >= 0):
        return 'prog'
    if np.all(diffs <= 0):
        return 'decay'
    return 'mixed'

