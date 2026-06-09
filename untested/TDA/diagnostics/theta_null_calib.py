"""
theta_null_calib.py
===================
θ_null 月度标定：通过随机打乱信号时间顺序构造点云，
计算 100 次 H1 持久性的 95% 分位数，作为"无循环零假设"下的噪声水平。

输出：configs/theta_null.parquet
列：stock_id, month(YYYYMM), config_id, theta
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
from ph_factors.point_cloud import build_point_cloud
from ph_factors.landmark import maxmin_landmark
from ph_factors.ph_kernels import compute_ph
from ph_factors.registry import CONFIGS, CloudConfig

H5_BASE    = "/home/sharedriver/public/Level2"
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

N_SHUFFLE  = 100    # 随机打乱次数
QUANTILE   = 0.95
LANDMARK_M = 200    # 标定阶段用较小 M，加速


def _theta_one_config(
    cfg: CloudConfig,
    signals: Dict,
    tau: int,
    n_shuffle: int = N_SHUFFLE,
    seed_base: int = 0,
) -> float:
    """
    对单个配置计算 θ_null：打乱主信号 n_shuffle 次，取 H1 MAXPERS 的 95% 分位数。
    """
    # 取 FULL 段、无方向的原始点云
    cloud = build_point_cloud(cfg, "FULL", None, signals, tau)
    if len(cloud) < 10:
        return 0.0

    main_col = 0   # 延迟嵌入的第一列就是主信号
    max_pers_list = []

    rng = np.random.default_rng(seed_base)
    for _ in range(n_shuffle):
        shuffled = cloud.copy()
        rng.shuffle(shuffled[:, main_col])   # 只打乱主信号维，保留环境维结构
        seed = int(rng.integers(0, 2**31))
        try:
            _, idx, dist = maxmin_landmark(shuffled, M=LANDMARK_M, seed=seed)
            _, pd_h1 = compute_ph(dist, maxdim=1)
            if len(pd_h1) > 0:
                pers = pd_h1[:, 1] - pd_h1[:, 0]
                max_pers_list.append(float(pers.max()))
            else:
                max_pers_list.append(0.0)
        except Exception:
            max_pers_list.append(0.0)

    if not max_pers_list:
        return 0.0
    return float(np.quantile(max_pers_list, QUANTILE))


def calibrate_theta_stock_month(
    stock_id: str,
    dates: List[str],
    tau_map: Dict,          # {config_id: {tau_level: int}}
    h5_root: str = H5_BASE,
) -> Dict[str, float]:
    """
    对一只股票的一个月，标定所有配置的 θ_null。
    取月内第一个有效交易日的信号做标定（θ_null 变化慢，单日足够）。

    Returns: {config_id: theta}
    """
    for date in dates:
        h5_path = os.path.join(h5_root, date[:4], date, f"{stock_id}.h5")
        if not os.path.exists(h5_path):
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                order_df = pd.DataFrame(f["order/table"][:])
                tick_df  = pd.DataFrame(f["tick/table"][:])
                trade_df = pd.DataFrame(f["trade/table"][:])
            sigs = compute_all_signals(order_df, tick_df, date,
                                       trade_df=trade_df, stock_id=stock_id,
                                       norm_params=None)
        except Exception:
            continue

        result = {}
        for cfg in CONFIGS:
            cfg_tau = tau_map.get(cfg.id, {})
            tau = cfg_tau.get(cfg.tau_level, 10)
            seed = hash(f"{stock_id}{date}{cfg.id}") & 0xFFFFFFFF
            result[cfg.id] = _theta_one_config(cfg, sigs, tau, seed_base=seed)
        return result

    # 所有日期都失败
    return {cfg.id: 0.0 for cfg in CONFIGS}


def run_calibration(
    stock_ids: List[str],
    dates: List[str],
    tau_parquet: str = None,
    h5_root: str = H5_BASE,
    output_path: str = None,
) -> pd.DataFrame:
    """
    对 stock_ids × 月份 标定 θ_null。

    输出列：stock_id, month, config_id, theta
    """
    if output_path is None:
        output_path = os.path.join(CONFIG_DIR, "theta_null.parquet")

    # 加载 tau 标定结果
    tau_df = None
    if tau_parquet and os.path.exists(tau_parquet):
        tau_df = pd.read_parquet(tau_parquet)

    def _get_tau_map(sid, month):
        if tau_df is None:
            return {}
        sub = tau_df[(tau_df["stock_id"] == sid) & (tau_df["month"] == month)]
        out = defaultdict(dict)
        for row in sub.itertuples(index=False):
            out[row.config_id][row.tau_level] = row.tau_ticks
        return dict(out)

    month_dates: Dict[str, List[str]] = defaultdict(list)
    for d in dates:
        month_dates[d[:6]].append(d)

    rows = []
    total = len(stock_ids) * len(month_dates)
    done  = 0

    for sid in stock_ids:
        for month, mdates in sorted(month_dates.items()):
            tau_map = _get_tau_map(sid, month)
            theta_map = calibrate_theta_stock_month(sid, mdates, tau_map, h5_root)
            for config_id, theta in theta_map.items():
                rows.append({
                    "stock_id":  sid,
                    "month":     month,
                    "config_id": config_id,
                    "theta":     theta,
                })
            done += 1
            if done % 50 == 0:
                print(f"  θ_null 标定进度: {done}/{total}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[OK] theta_null 写出: {output_path}  shape={df.shape}")
    return df


def _cli():
    p = argparse.ArgumentParser(description="TDA θ_null 月度标定")
    p.add_argument("--stock_list",   type=str, required=True)
    p.add_argument("--start_date",   type=str, required=True)
    p.add_argument("--end_date",     type=str, required=True)
    p.add_argument("--tau_parquet",  type=str, default=None,
                   help="tau_calibration.parquet 路径（可选，无则用默认 τ=10）")
    p.add_argument("--h5_root",      type=str, default=H5_BASE)
    p.add_argument("--output",       type=str, default=None)
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

    run_calibration(stock_ids, dates, args.tau_parquet, args.h5_root, args.output)


if __name__ == "__main__":
    _cli()
