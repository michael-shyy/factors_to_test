"""refill_tracker.py — 补给速度追踪（向量化）。"""
import numpy as np
import pandas as pd


def compute_refill_speed(
    events: pd.DataFrame,
    tick: pd.DataFrame,
    max_wait_sec: float = 30.0,
    tol: float = 0.2,
) -> dict[str, float]:
    if len(events) == 0:
        return {'resupply_bid': np.nan, 'resupply_ask': np.nan, 'COLLAPSE_ASYM': np.nan}

    tick_s = tick.sort_values('time').reset_index(drop=True)
    tick_times = tick_s['time'].values
    t_fill = 1.0 / max_wait_sec

    speeds = {'bid': [], 'ask': []}

    for side, vol_col in [('bid', 'bdv1'), ('ask', 'akv1')]:
        if vol_col not in tick_s.columns:
            continue
        vol_arr = tick_s[vol_col].values
        ev_side = events[events['side'] == side]
        if len(ev_side) == 0:
            continue

        t_events = ev_side['time'].values
        # 每个事件的起始 tick 索引
        ti_arr = np.searchsorted(tick_times, t_events, side='right')
        # 每个事件的搜索上界（不超过 max_wait_sec）
        ti_end = np.searchsorted(tick_times, t_events + max_wait_sec, side='right')

        for k in range(len(t_events)):
            ti = ti_arr[k]
            if ti == 0 or ti >= len(tick_times):
                speeds[side].append(t_fill)
                continue
            level_before = vol_arr[ti - 1]
            if level_before <= 0:
                continue

            # 向量化：在 [ti, ti_end[k]) 范围内找第一个满足恢复条件的 tick
            seg_vol = vol_arr[ti:ti_end[k]]
            seg_t   = tick_times[ti:ti_end[k]]
            if len(seg_vol) == 0:
                speeds[side].append(t_fill)
                continue
            recovered_mask = np.abs(seg_vol - level_before) / level_before <= tol
            idx = np.argmax(recovered_mask)  # 第一个 True 的位置
            if recovered_mask[idx]:
                dt = max(seg_t[idx] - t_events[k], 1e-3)
                speeds[side].append(1.0 / dt)
            else:
                speeds[side].append(t_fill)

    def _m(lst): return float(np.mean(lst)) if lst else np.nan
    rb = _m(speeds['bid'])
    ra = _m(speeds['ask'])
    asym = float(rb - ra) if not (np.isnan(rb) or np.isnan(ra)) else np.nan
    return {'resupply_bid': rb, 'resupply_ask': ra, 'COLLAPSE_ASYM': asym}

