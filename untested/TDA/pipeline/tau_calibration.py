"""
tau_calibration.py
==================
τ 月度标定脚本。

三档 τ 定义（单位：3 秒桶数）：
  short : 主信号自相关首次过零位置 / 2
  mid   : 主信号自相关首次过零位置
  long  : 主信号互信息函数首次极小位置

输出：configs/tau_calibration.parquet
列：stock_id, month(YYYYMM), config_id, tau_level, tau_ticks
"""

# ── sys.path bootstrap ──────────────────────────────────────────────────────
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, _os.path.join(_PROJECT_ROOT, "ph_factors"), "/home/hysheng"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ───────────────────────────────────────────────────────────────────────────

import os
import glob
import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import h5py

try:
    import hdf5plugin  # noqa
    os.environ["HDF5_PLUGIN_PATH"] = (
        "/home/hysheng/.local/lib/python3.10/site-packages/hdf5plugin/plugins/"
    )
except ImportError:
    pass

from ph_factors.signals import compute_all_signals
from ph_factors.registry import CONFIGS

H5_BASE    = "/home/sharedriver/public/Level2"
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

# 主信号名（去掉方向后缀）
_MAIN_SIGNALS = list({cfg.main_signal for cfg in CONFIGS})

# τ 搜索上限（桶数）
_TAU_MAX = 120   # 120 × 3s = 6 分钟


# ── 自相关首次过零 ────────────────────────────────────────────────────────────

def _acf_first_zero(arr: np.ndarray, max_lag: int = _TAU_MAX) -> int:
    """返回自相关函数首次过零的 lag（桶数），找不到则返回 max_lag。"""
    arr = arr - arr.mean()
    std = arr.std()
    if std < 1e-10:
        return max_lag
    arr = arr / std
    n = len(arr)
    prev_acf = 1.0
    for lag in range(1, min(max_lag + 1, n)):
        acf = float(np.dot(arr[:-lag], arr[lag:]) / (n - lag))
        if acf <= 0 and prev_acf > 0:
            return lag
        prev_acf = acf
    return max_lag


# ── 互信息首次极小 ────────────────────────────────────────────────────────────

def _mi_first_min(arr: np.ndarray, max_lag: int = _TAU_MAX, bins: int = 20) -> int:
    """返回互信息函数首次极小的 lag（桶数），找不到则返回 max_lag。"""
    arr = arr - arr.min()
    rng = arr.max()
    if rng < 1e-10:
        return max_lag
    arr = arr / rng  # 归一化到 [0,1]

    def _mi(lag):
        x, y = arr[:-lag], arr[lag:]
        hist2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist2d / hist2d.sum()
        px  = pxy.sum(axis=1, keepdims=True)
        py  = pxy.sum(axis=0, keepdims=True)
        mask = pxy > 0
        return float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py + 1e-12)[mask])))

    prev_mi = _mi(1)
    for lag in range(2, min(max_lag + 1, len(arr))):
        mi = _mi(lag)
        if mi > prev_mi:          # 极小：前一个 lag 是局部最小
            return lag - 1
        prev_mi = mi
    return max_lag


# ── 单股单月标定 ──────────────────────────────────────────────────────────────

def calibrate_stock_month(
    stock_id: str,
    dates: List[str],
    h5_root: str = H5_BASE,
) -> Dict:
    """
    对一只股票的一个月内所有交易日，计算各主信号的 τ 三档。

    Returns:
        {signal_name: {'short': int, 'mid': int, 'long': int}}
    """
    signal_arrays: Dict[str, List[np.ndarray]] = defaultdict(list)

    for date in dates:
        h5_path = os.path.join(h5_root, date[:4], date, f"{stock_id}.h5")
        if not os.path.exists(h5_path):
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                order_df = pd.DataFrame(f["order/table"][:])
                tick_df  = pd.DataFrame(f["tick/table"][:])
                trade_df = pd.DataFrame(f["trade/table"][:])
            # 不做标准化，只要原始序列
            sigs = compute_all_signals(order_df, tick_df, date,
                                       trade_df=trade_df, stock_id=stock_id,
                                       norm_params=None)
            for sig_name in _MAIN_SIGNALS:
                s = sigs.get(sig_name)
                if s is not None and len(s) > 20:
                    signal_arrays[sig_name].append(s.values.astype(np.float64))
        except Exception:
            continue

    result = {}
    for sig_name in _MAIN_SIGNALS:
        arrays = signal_arrays.get(sig_name, [])
        if not arrays:
            result[sig_name] = {"short": 10, "mid": 20, "long": 40}
            continue
        combined = np.concatenate(arrays)
        mid_tau   = max(1, _acf_first_zero(combined))
        short_tau = max(1, mid_tau // 2)
        long_tau  = max(mid_tau + 1, _mi_first_min(combined))
        result[sig_name] = {"short": short_tau, "mid": mid_tau, "long": long_tau}

    return result


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run_calibration(
    stock_ids: List[str],
    dates: List[str],          # 全部交易日，函数内按月分组
    h5_root: str = H5_BASE,
    output_path: str = None,
) -> pd.DataFrame:
    """
    对 stock_ids × 月份 标定 τ，返回并保存 DataFrame。

    输出列：stock_id, month, config_id, tau_level, tau_ticks
    """
    if output_path is None:
        output_path = os.path.join(CONFIG_DIR, "tau_calibration.parquet")

    # 按月分组
    month_dates: Dict[str, List[str]] = defaultdict(list)
    for d in dates:
        month_dates[d[:6]].append(d)

    rows = []
    total = len(stock_ids) * len(month_dates)
    done  = 0
    for sid in stock_ids:
        for month, mdates in sorted(month_dates.items()):
            tau_map = calibrate_stock_month(sid, mdates, h5_root)
            # 展开到 config 级别
            for cfg in CONFIGS:
                sig_tau = tau_map.get(cfg.main_signal, {"short": 10, "mid": 20, "long": 40})
                for level, ticks in sig_tau.items():
                    rows.append({
                        "stock_id":  sid,
                        "month":     month,
                        "config_id": cfg.id,
                        "tau_level": level,
                        "tau_ticks": ticks,
                    })
            done += 1
            if done % 50 == 0:
                print(f"  τ 标定进度: {done}/{total}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[OK] tau_calibration 写出: {output_path}  shape={df.shape}")
    return df


def _cli():
    p = argparse.ArgumentParser(description="TDA τ 月度标定")
    p.add_argument("--stock_list", type=str, required=True,
                   help="股票代码文件（每行一个，如 sh600000）")
    p.add_argument("--start_date", type=str, required=True)
    p.add_argument("--end_date",   type=str, required=True)
    p.add_argument("--h5_root",    type=str, default=H5_BASE)
    p.add_argument("--output",     type=str, default=None)
    args = p.parse_args()

    with open(args.stock_list) as f:
        stock_ids = [l.strip() for l in f if l.strip()]

    # 扫交易日
    dates = []
    for year in sorted(os.listdir(args.h5_root)):
        if not year.isdigit():
            continue
        ydir = os.path.join(args.h5_root, year)
        for d in sorted(os.listdir(ydir)):
            if d.isdigit() and len(d) == 8 and args.start_date <= d <= args.end_date:
                dates.append(d)

    run_calibration(stock_ids, dates, args.h5_root, args.output)


if __name__ == "__main__":
    _cli()
