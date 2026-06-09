"""
probe_lz.py — 探价前置标记版本 LZ。

利用 shared_features.probe_flag，将大单序列拆分为
探价前置型（probe_flag=1）和突发型（probe_flag=0），分别计算 LZ76。

时间戳规则：probe_flag 由预处理阶段计算，t 时刻标记只依赖 ≤t-1 的历史，无前视。
"""
import numpy as np
from .lz76 import lz76_normalized


def probe_lz_factors(
    seq: np.ndarray,
    bs_flag: np.ndarray,
    volume: np.ndarray,
    probe_flag: np.ndarray,
    window: int = 500,
    p_large: float = 0.8,
) -> dict[str, float]:
    """
    对探价前置型和突发型大单子序列分别计算 LZ76。

    Parameters
    ----------
    seq        : 二值或六符号编码序列
    bs_flag    : 买卖方向（0=买）
    volume     : 成交量
    probe_flag : 探价前置标记（1=探价，0=突发），与 seq 等长
    window     : LZ 窗口
    p_large    : 大单阈值分位数

    Returns
    -------
    dict 含 LZ_probe, LZ_burst_type, LZ_probe_diff
    """
    q_large = np.nanquantile(volume, p_large)
    large_mask = volume >= q_large

    probe_mask = large_mask & (probe_flag == 1)
    burst_mask = large_mask & (probe_flag == 0)

    def _lz(mask: np.ndarray) -> float:
        sub = seq[mask]
        if len(sub) < window // 4:
            return np.nan
        return lz76_normalized(sub[-window:])

    lz_probe = _lz(probe_mask)
    lz_burst = _lz(burst_mask)
    diff = (lz_probe - lz_burst
            if not (np.isnan(lz_probe) or np.isnan(lz_burst)) else np.nan)

    return {
        'LZ_probe':      lz_probe,
        'LZ_burst_type': lz_burst,
        'LZ_probe_diff': diff,
    }
