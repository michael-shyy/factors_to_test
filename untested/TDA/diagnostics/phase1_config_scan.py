"""
phase1_config_scan.py
=====================
阶段 1 配置粗扫：对 14 个配置只算 GEO_PH_{CONFIG}_H1_MAXPERS_FULL，
输出诊断报告 + 胜出配置清单。

决策规则（spec §7）：
  - MAXPERS 均值 < 1.5 × θ_null → 废配置
  - MAXPERS 跨日 IQR / median > 2 → 不稳定
  - 同簇内相关性 > 0.95 → 留 alpha 更强的一个
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
from typing import List, Dict

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
from ph_factors.pd_statistics import compute_pd_stats
from ph_factors.registry import CONFIGS

H5_BASE    = "/home/sharedriver/public/Level2"
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
LANDMARK_M = 500


def _scan_one_stock_day(
    stock_id: str,
    date: str,
    tau_map: Dict,      # {config_id: {tau_level: int}}
    theta_map: Dict,    # {config_id: float}
    h5_root: str,
) -> Dict[str, float]:
    """
    对单股单日计算所有配置的 H1_MAXPERS_FULL。
    Returns: {config_id: maxpers}
    """
    h5_path = os.path.join(h5_root, date[:4], date, f"{stock_id}.h5")
    if not os.path.exists(h5_path):
        return {}
    try:
        with h5py.File(h5_path, "r") as f:
            order_df = pd.DataFrame(f["order/table"][:])
            tick_df  = pd.DataFrame(f["tick/table"][:])
            trade_df = pd.DataFrame(f["trade/table"][:])
        sigs = compute_all_signals(order_df, tick_df, date,
                                   trade_df=trade_df, stock_id=stock_id,
                                   norm_params=None)
    except Exception:
        return {}

    result = {}
    for cfg in CONFIGS:
        cfg_tau = tau_map.get(cfg.id, {})
        tau     = cfg_tau.get(cfg.tau_level, 10)
        theta   = theta_map.get(cfg.id, 0.0)
        try:
            cloud = build_point_cloud(cfg, "FULL", None, sigs, tau)
            if len(cloud) < 4:
                result[cfg.id] = float("nan")
                continue
            seed = hash(f"{stock_id}{date}{cfg.id}") & 0xFFFFFFFF
            _, _, dist = maxmin_landmark(cloud, M=LANDMARK_M, seed=seed)
            _, pd_h1   = compute_ph(dist, maxdim=1)
            stats = compute_pd_stats(pd_h1, theta)
            result[cfg.id] = stats["MAXPERS"]
        except Exception:
            result[cfg.id] = float("nan")
    return result


def run_phase1_scan(
    stock_ids: List[str],
    dates: List[str],
    tau_parquet: str = None,
    theta_parquet: str = None,
    h5_root: str = H5_BASE,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    对 stock_ids × dates 跑配置粗扫，输出诊断报告。

    Returns: DataFrame，index=config_id，列包含诊断指标和决策。
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "diagnostics", "phase1_results",
        )
    os.makedirs(output_dir, exist_ok=True)

    # 加载标定结果
    def _load_parquet(path):
        return pd.read_parquet(path) if path and os.path.exists(path) else None

    tau_df   = _load_parquet(tau_parquet)
    theta_df = _load_parquet(theta_parquet)

    def _get_tau(sid, date):
        if tau_df is None:
            return {}
        month = date[:6]
        sub = tau_df[(tau_df["stock_id"] == sid) & (tau_df["month"] == month)]
        out = defaultdict(dict)
        for row in sub.itertuples(index=False):
            out[row.config_id][row.tau_level] = row.tau_ticks
        return dict(out)

    def _get_theta(sid, date):
        if theta_df is None:
            return {}
        month = date[:6]
        sub = theta_df[(theta_df["stock_id"] == sid) & (theta_df["month"] == month)]
        return {row.config_id: row.theta for row in sub.itertuples(index=False)}

    # 收集所有 (stock, date, config) 的 MAXPERS
    records = []
    total = len(stock_ids) * len(dates)
    done  = 0
    for sid in stock_ids:
        for date in dates:
            tau_map   = _get_tau(sid, date)
            theta_map = _get_theta(sid, date)
            day_result = _scan_one_stock_day(sid, date, tau_map, theta_map, h5_root)
            for config_id, maxpers in day_result.items():
                records.append({
                    "stock_id":  sid,
                    "date":      date,
                    "config_id": config_id,
                    "maxpers":   maxpers,
                })
            done += 1
            if done % 100 == 0:
                print(f"  扫描进度: {done}/{total}", flush=True)

    raw_df = pd.DataFrame(records)
    raw_df.to_parquet(os.path.join(output_dir, "phase1_raw.parquet"), index=False)

    # 加载 theta 均值（按 config 聚合）
    theta_mean: Dict[str, float] = {}
    if theta_df is not None:
        theta_mean = theta_df.groupby("config_id")["theta"].mean().to_dict()

    # 诊断指标
    diag_rows = []
    for config_id, grp in raw_df.groupby("config_id"):
        vals = grp["maxpers"].dropna()
        if len(vals) == 0:
            continue
        mean_mp  = float(vals.mean())
        median_mp = float(vals.median())
        q75, q25 = float(vals.quantile(0.75)), float(vals.quantile(0.25))
        iqr_ratio = (q75 - q25) / (median_mp + 1e-9)
        theta     = theta_mean.get(config_id, 0.0)

        # 决策规则
        kill_weak     = (theta > 0) and (mean_mp < 1.5 * theta)
        kill_unstable = iqr_ratio > 2.0
        survive = not (kill_weak or kill_unstable)

        diag_rows.append({
            "config_id":   config_id,
            "cluster":     next((c.cluster for c in CONFIGS if c.id == config_id), "?"),
            "mean_maxpers": round(mean_mp, 4),
            "median_maxpers": round(median_mp, 4),
            "iqr_ratio":   round(iqr_ratio, 3),
            "theta_null":  round(theta, 4),
            "ratio_to_theta": round(mean_mp / (theta + 1e-9), 2),
            "kill_weak":   kill_weak,
            "kill_unstable": kill_unstable,
            "survive_prelim": survive,
        })

    diag_df = pd.DataFrame(diag_rows).set_index("config_id")

    # 簇内相关性过滤（同簇 survive 的配置间相关 > 0.95 → 留 mean_maxpers 最大的）
    pivot = raw_df.pivot_table(index=["stock_id", "date"], columns="config_id", values="maxpers")
    survivors = diag_df[diag_df["survive_prelim"]].index.tolist()
    diag_df["survive_final"] = diag_df["survive_prelim"]

    clusters = diag_df.loc[survivors, "cluster"].unique()
    for cluster in clusters:
        cluster_cfgs = diag_df[(diag_df["cluster"] == cluster) & diag_df["survive_prelim"]].index.tolist()
        if len(cluster_cfgs) < 2:
            continue
        sub_pivot = pivot[cluster_cfgs].dropna()
        if len(sub_pivot) < 5:
            continue
        corr = sub_pivot.corr()
        # 贪心：按 mean_maxpers 降序，若与已保留的相关 > 0.95 则剔除
        ranked = diag_df.loc[cluster_cfgs, "mean_maxpers"].sort_values(ascending=False).index.tolist()
        kept = []
        for cfg in ranked:
            if all(abs(corr.loc[cfg, k]) <= 0.95 for k in kept):
                kept.append(cfg)
            else:
                diag_df.loc[cfg, "survive_final"] = False

    # 输出报告
    report_path = os.path.join(output_dir, "phase1_report.csv")
    diag_df.to_csv(report_path)

    winners = diag_df[diag_df["survive_final"]].index.tolist()
    winner_path = os.path.join(output_dir, "winning_configs.txt")
    with open(winner_path, "w") as f:
        f.write("\n".join(winners) + "\n")

    print(f"\n{'='*60}")
    print(f"阶段 1 配置粗扫结果")
    print(f"{'='*60}")
    print(diag_df[["cluster", "mean_maxpers", "ratio_to_theta",
                   "iqr_ratio", "kill_weak", "kill_unstable", "survive_final"]].to_string())
    print(f"\n胜出配置（{len(winners)} 个）: {winners}")
    print(f"报告: {report_path}")
    print(f"胜出清单: {winner_path}")

    return diag_df


def _cli():
    p = argparse.ArgumentParser(description="TDA 阶段 1 配置粗扫")
    p.add_argument("--stock_list",    type=str, required=True)
    p.add_argument("--start_date",    type=str, required=True)
    p.add_argument("--end_date",      type=str, required=True)
    p.add_argument("--tau_parquet",   type=str, default=None)
    p.add_argument("--theta_parquet", type=str, default=None)
    p.add_argument("--h5_root",       type=str, default=H5_BASE)
    p.add_argument("--output_dir",    type=str, default=None)
    args = p.parse_args()

    with open(args.stock_list) as f:
        stock_ids = [l.strip() for l in f if l.strip()]

    dates = []
    for year in sorted(os.listdir(args.h5_root)):
        if not year.isdigit():
            continue
        for d in sorted(os.listdir(os.path.join(args.h5_root, year))):
            if d.isdigit() and len(d) == 8 and args.start_date <= d <= args.end_date:
                dates.append(d)

    run_phase1_scan(
        stock_ids, dates,
        tau_parquet=args.tau_parquet,
        theta_parquet=args.theta_parquet,
        h5_root=args.h5_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    _cli()
