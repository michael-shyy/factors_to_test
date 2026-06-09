"""
norm_calibration.py
===================
信号标准化参数(z-score 的 mean/std)月度标定脚本 —— 这是 signals.py 做
股票×月度 z-score 所必需的表，缺它则 z-score 不生效（所有信号保持原始量级，
OFI ~1e5、失衡 ~1e-2 差几个数量级，PD 完全不可比）。

对一只股票一个月内所有交易日，把每个信号的全部样本拼起来，算 mean / std。

输出：configs/norm_params.parquet
列：stock_id, month(YYYYMM), signal_name, mean, std
"""

# ── sys.path bootstrap ──────────────────────────────────────────────────────
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, _os.path.join(_PROJECT_ROOT, "ph_factors"), "/home/hysheng"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ───────────────────────────────────────────────────────────────────────────

import os
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

H5_BASE    = "/home/sharedriver/public/Level2"
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

# 需要标准化的全部信号（主信号 + 方向化 + 环境维）
_SIGNAL_NAMES = [
    "OFI_NET", "OFI_NET_BUY", "OFI_NET_SELL",
    "OB_IMBAL", "OB_IMBAL_BUY", "OB_IMBAL_SELL",
    "DP_MICRO",
    "CANCEL_NET", "CANCEL_NET_BUY", "CANCEL_NET_SELL",
    "SPREAD", "DEPTH5", "RV_PROXY", "CANCEL_RATE", "LIFE_TIME",
]


def calibrate_stock_month(
    stock_id: str,
    dates: List[str],
    h5_root: str = H5_BASE,
) -> Dict[str, Dict[str, float]]:
    """
    对一只股票一个月内所有交易日，累积每个信号的样本，算 mean/std。

    Returns: {signal_name: {'mean': float, 'std': float}}
    """
    pools: Dict[str, List[np.ndarray]] = defaultdict(list)

    for date in dates:
        h5_path = os.path.join(h5_root, date[:4], date, f"{stock_id}.h5")
        if not os.path.exists(h5_path):
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                order_df = pd.DataFrame(f["order/table"][:])
                tick_df  = pd.DataFrame(f["tick/table"][:])
                trade_df = pd.DataFrame(f["trade/table"][:])
            # 不标准化，取原始序列
            sigs = compute_all_signals(order_df, tick_df, date,
                                       trade_df=trade_df, stock_id=stock_id,
                                       norm_params=None)
            for name in _SIGNAL_NAMES:
                s = sigs.get(name)
                if s is not None and len(s) > 0:
                    vals = s.values.astype(np.float64)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) > 0:
                        pools[name].append(vals)
        except Exception:
            continue

    result = {}
    for name in _SIGNAL_NAMES:
        arrays = pools.get(name, [])
        if not arrays:
            # 缺数据时给 mean=0 std=1（等价于不标准化但不会 NaN）
            result[name] = {"mean": 0.0, "std": 1.0}
            continue
        combined = np.concatenate(arrays)
        mean = float(np.mean(combined))
        std  = float(np.std(combined))
        if std < 1e-10:
            std = 1.0   # 常数信号，避免除零
        result[name] = {"mean": mean, "std": std}

    return result


def run_calibration(
    stock_ids: List[str],
    dates: List[str],
    h5_root: str = H5_BASE,
    output_path: str = None,
) -> pd.DataFrame:
    """对 stock_ids × 月份 标定 mean/std，返回并保存 DataFrame。"""
    if output_path is None:
        output_path = os.path.join(CONFIG_DIR, "norm_params.parquet")

    month_dates: Dict[str, List[str]] = defaultdict(list)
    for d in dates:
        month_dates[d[:6]].append(d)

    rows = []
    total = len(stock_ids) * len(month_dates)
    done = 0
    for sid in stock_ids:
        for month, mdates in sorted(month_dates.items()):
            norm_map = calibrate_stock_month(sid, mdates, h5_root)
            for sig_name, ms in norm_map.items():
                rows.append({
                    "stock_id":    sid,
                    "month":       month,
                    "signal_name": sig_name,
                    "mean":        ms["mean"],
                    "std":         ms["std"],
                })
            done += 1
            if done % 50 == 0:
                print(f"  norm 标定进度: {done}/{total}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[OK] norm_params 写出: {output_path}  shape={df.shape}")
    return df


def _cli():
    p = argparse.ArgumentParser(description="TDA 信号标准化参数月度标定")
    p.add_argument("--stock_list", type=str, required=True)
    p.add_argument("--start_date", type=str, required=True)
    p.add_argument("--end_date",   type=str, required=True)
    p.add_argument("--h5_root",    type=str, default=H5_BASE)
    p.add_argument("--output",     type=str, default=None)
    args = p.parse_args()

    with open(args.stock_list) as f:
        stock_ids = [l.strip() for l in f if l.strip()]

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
