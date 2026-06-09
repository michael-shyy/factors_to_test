"""
price_context.py — 价格背景条件化 LZ。

价格背景定义（均基于 running VWAP，严禁用全日 VWAP）
-------------------------------------------------
upper  : 中间价 > running_vwap + 0.5 * running_sigma
lower  : 中间价 < running_vwap - 0.5 * running_sigma
vwap   : 中间价 ∈ [running_vwap - 0.3σ, running_vwap + 0.3σ]
new_high: 创日内新高后的 50 笔

时间戳规则：running_vwap/sigma 均为截至 t-1 时刻的累计值，无前视。
"""
import numpy as np
import pandas as pd
from .lz76 import lz76_normalized


def _running_vwap_sigma(prices: np.ndarray, volumes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    running VWAP 和 rolling sigma（5分钟窗口，shift(1)）。
    返回与 prices 等长的 (vwap, sigma) 数组，前若干位为 nan。
    """
    n = len(prices)
    cum_pv = np.cumsum(prices * volumes)
    cum_v  = np.cumsum(volumes)

    # shift(1)：t 时刻 VWAP 用截至 t-1 的累计值
    vwap = np.empty(n)
    vwap[0] = np.nan
    vwap[1:] = np.where(cum_v[:-1] > 0, cum_pv[:-1] / cum_v[:-1], np.nan)

    # rolling std（窗口约 75 笔 ≈ 5 分钟 @4s/tick）用 shift(1)
    WINDOW = 75
    # ddof=0 匹配旧版 numpy.std 行为
    sigma = pd.Series(prices).shift(1).rolling(WINDOW, min_periods=WINDOW).std(ddof=0).values

    return vwap, sigma


def price_context_masks(
    prices: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    返回各价格背景的布尔掩码（与 prices 等长）。
    key: 'upper', 'lower', 'vwap_range', 'new_high_50'
    """
    vwap, sigma = _running_vwap_sigma(prices, volumes)

    upper = prices > vwap + 0.5 * sigma
    lower = prices < vwap - 0.5 * sigma
    vwap_range = (prices >= vwap - 0.3 * sigma) & (prices <= vwap + 0.3 * sigma)

    # 创日内新高标记：当前价格 > 截至此刻的历史最高价（用 shift(1) 取历史最高）
    running_max = np.maximum.accumulate(prices)
    shifted_max = np.empty(len(prices))
    shifted_max[0] = -np.inf
    shifted_max[1:] = running_max[:-1]
    new_high_flag = prices > shifted_max

    # 新高后 50 笔
    new_high_50 = np.zeros(len(prices), dtype=bool)
    indices = np.where(new_high_flag)[0]
    for idx in indices:
        end = min(idx + 51, len(prices))
        new_high_50[idx:end] = True

    return {
        'upper': upper,
        'lower': lower,
        'vwap_range': vwap_range,
        'new_high_50': new_high_50,
    }


def lz_price_context(
    seq: np.ndarray,
    masks: dict[str, np.ndarray],
    window: int = 500,
    alphabet_size: int = 2,
) -> dict[str, float]:
    """
    对各价格背景子序列计算末尾 window 笔的 LZ76。
    返回 LZ_upper, LZ_lower, LZ_vwap_regime, LZ_new_high_regime, LZ_price_context_diff。
    """
    def _lz(mask: np.ndarray) -> float:
        sub = seq[mask]
        if len(sub) < window // 4:
            return np.nan
        return lz76_normalized(sub[-window:], alphabet_size)

    lz_upper  = _lz(masks['upper'])
    lz_lower  = _lz(masks['lower'])
    lz_vwap   = _lz(masks['vwap_range'])
    lz_newhigh = _lz(masks['new_high_50'])

    diff = (lz_upper - lz_lower
            if not (np.isnan(lz_upper) or np.isnan(lz_lower)) else np.nan)

    return {
        'LZ_upper':             lz_upper,
        'LZ_lower':             lz_lower,
        'LZ_vwap_regime':       lz_vwap,
        'LZ_new_high_regime':   lz_newhigh,
        'LZ_price_context_diff': diff,
    }
