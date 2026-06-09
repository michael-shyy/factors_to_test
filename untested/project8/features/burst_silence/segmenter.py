"""segmenter.py — 爆发/静默段识别（numba 加速状态机）。"""
import numpy as np
from dataclasses import dataclass, field
from numba import njit


@dataclass
class Segment:
    kind: str
    start: float
    end: float
    n: int
    vol_seq: np.ndarray = field(default_factory=lambda: np.empty(0))
    bs_seq:  np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))


@njit(cache=True)
def _state_machine(is_burst_gap: np.ndarray, is_silence_gap: np.ndarray) -> np.ndarray:
    """顺序依赖状态机，numba JIT 消掉 Python 循环开销。"""
    n = len(is_burst_gap)
    state = np.empty(n, dtype=np.int8)
    state[0] = np.int8(0) if is_burst_gap[0] else np.int8(1)
    for i in range(1, n):
        if state[i-1] == 0:
            state[i] = np.int8(0) if is_burst_gap[i] else np.int8(1)
        else:
            state[i] = np.int8(1) if is_silence_gap[i] else np.int8(0)
    return state


def segment_trades(
    times: np.ndarray,
    volumes: np.ndarray,
    bs_flags: np.ndarray,
    p_burst: float = 0.25,
    p_silence: float = 0.75,
) -> list[Segment]:
    if len(times) < 3:
        return []

    dt = np.diff(times)
    tau_burst   = np.nanquantile(dt, p_burst)
    tau_silence = np.nanquantile(dt, p_silence)

    state = _state_machine(dt <= tau_burst, dt >= tau_silence)

    trade_kinds = np.empty(len(times), dtype=np.int8)
    trade_kinds[0] = state[0]
    trade_kinds[1:] = state

    changes = np.flatnonzero(np.diff(trade_kinds))
    starts  = np.concatenate([[0], changes + 1])
    ends    = np.concatenate([changes + 1, [len(times)]])

    vols_c = np.ascontiguousarray(volumes)
    bs_c   = np.ascontiguousarray(bs_flags, dtype=np.int8)

    return [
        Segment(
            kind='burst' if trade_kinds[s] == 0 else 'silence',
            start=float(times[s]),
            end=float(times[e - 1]),
            n=int(e - s),
            vol_seq=vols_c[s:e],
            bs_seq=bs_c[s:e],
        )
        for s, e in zip(starts, ends)
    ]


def split_by_direction(
    times: np.ndarray,
    volumes: np.ndarray,
    bs_flags: np.ndarray,
    target_bs: int,
    **kw,
) -> list[Segment]:
    mask = bs_flags == target_bs
    if mask.sum() < 3:
        return []
    return segment_trades(times[mask], volumes[mask], bs_flags[mask], **kw)
