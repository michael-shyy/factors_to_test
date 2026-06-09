"""
time_one_stock.py
=================
测单股单日的实际耗时，分阶段打印，定位瓶颈。

用法：
    # 自动拼路径（推荐）
    python tools/time_one_stock.py sh600000 20250507

    # 直接给 h5 路径
    python tools/time_one_stock.py --h5 /path/to/sh600000.h5 --date 20250507

    # 带标定表（测真实生产耗时；不带则用默认 τ=10、不标准化，耗时结构基本一致）
    python tools/time_one_stock.py sh600000 20250507 --calib_dir configs

说明：
    - 第一次跑含 numba JIT 编译开销，脚本会先 warmup 再正式计时，
      所以打印的耗时是"热"状态下的真实单股耗时。
    - 耗时主要由 ripser 决定，标定表只影响 τ（轻微改变点云大小），
      不带标定表测出的总耗时和真实生产相差很小。
"""
# ── sys.path bootstrap ──────────────────────────────────────────────────────
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, _os.path.join(_PROJECT_ROOT, "ph_factors"), "/home/hysheng"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ───────────────────────────────────────────────────────────────────────────

import os
import time
import argparse
from collections import defaultdict

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

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_ph6")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

H5_BASE = "/home/sharedriver/public/Level2"


def _load_h5(file_path: str):
    with h5py.File(file_path, "r") as f:
        order_df = pd.DataFrame(f["order/table"][:])
        tick_df  = pd.DataFrame(f["tick/table"][:])
        trade_df = pd.DataFrame(f["trade/table"][:])
    for df in [order_df, tick_df, trade_df]:
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = df[col].str.decode("utf-8")
            except AttributeError:
                pass
    return order_df, tick_df, trade_df


def _load_calib(calib_dir, month, stock_id):
    """加载该股该月的 tau/norm/theta，返回 (tau_lookup, norm_params, theta_lookup)。"""
    def _read(name):
        p = os.path.join(calib_dir, f"{name}.parquet")
        return pd.read_parquet(p) if os.path.exists(p) else None

    tau_df   = _read("tau_calibration")
    norm_df  = _read("norm_params")
    theta_df = _read("theta_null")

    tau_lookup = {}
    if tau_df is not None:
        sub = tau_df[(tau_df["stock_id"] == stock_id) & (tau_df["month"] == month)]
        for r in sub.itertuples(index=False):
            tau_lookup.setdefault(r.config_id, {})[r.tau_level] = r.tau_ticks

    norm_params = {}
    if norm_df is not None:
        sub = norm_df[(norm_df["stock_id"] == stock_id) & (norm_df["month"] == month)]
        for r in sub.itertuples(index=False):
            norm_params[r.signal_name] = {"mean": r.mean, "std": r.std}

    theta_lookup = {}
    if theta_df is not None:
        sub = theta_df[(theta_df["stock_id"] == stock_id) & (theta_df["month"] == month)]
        for r in sub.itertuples(index=False):
            theta_lookup[r.config_id] = r.theta

    return tau_lookup, norm_params, theta_lookup


def time_one_stock(file_path, date_str, calib_dir=None, verbose=True):
    from ph_factors.signals import compute_all_signals
    from ph_factors.point_cloud import build_point_cloud
    from ph_factors.landmark import maxmin_landmark
    from ph_factors.ph_kernels import compute_ph
    from ph_factors.pd_statistics import compute_pd_stats
    from ph_factors.registry import CONFIGS, ALL_FACTOR_SPECS, PointCloudCache

    stock_id = os.path.basename(file_path).replace(".h5", "")
    month = date_str[:6]

    # ── numba warmup（不计入正式计时）──
    maxmin_landmark(np.random.default_rng(0).random((50, 2)), M=20, seed=0)

    # ── 标定表 ──
    if calib_dir:
        tau_lookup, norm_params, theta_lookup = _load_calib(calib_dir, month, stock_id)
    else:
        tau_lookup, norm_params, theta_lookup = {}, {}, {}

    timings = {}

    # ── 1. 数据加载 ──
    t0 = time.time()
    order_df, tick_df, trade_df = _load_h5(file_path)
    timings["1_load_h5"] = time.time() - t0

    # ── 2. 信号计算 ──
    t0 = time.time()
    signals = compute_all_signals(
        order_df, tick_df, date_str,
        trade_df=trade_df, stock_id=stock_id,
        norm_params=norm_params or None,
    )
    timings["2_signals"] = time.time() - t0

    # ── 3. 点云 + PD（分配置统计）──
    t_pc = t_lm = t_ph = 0.0
    pd_count = 0
    per_config = defaultdict(float)
    cache = {}

    t_total_pd = time.time()
    for cfg in CONFIGS:
        cfg_tau  = tau_lookup.get(cfg.id, {})
        theta    = theta_lookup.get(cfg.id, 0.0)
        h1_segs  = set(cfg.h1_segments)
        h0_segs  = set(cfg.h0_segments)
        all_segs = h1_segs | h0_segs
        for seg in all_segs:
            need_h1 = seg in h1_segs
            for direction in cfg.directions:
                key = (cfg.id, seg, direction)
                if key in cache:
                    continue
                tau = cfg_tau.get(cfg.tau_level, 10)
                is_dir = direction in ("BUY", "SELL")
                seg_need_h1 = need_h1 or is_dir
                seg_need_h0 = (seg in h0_segs) and (not is_dir)
                if not (seg_need_h1 or seg_need_h0):
                    continue

                cfg_t0 = time.time()

                tt = time.time()
                cloud = build_point_cloud(cfg, seg, direction, signals, tau)
                t_pc += time.time() - tt

                if len(cloud) < 4:
                    cache[key] = PointCloudCache(
                        config_id=cfg.id, segment=seg, direction=direction,
                        raw_cloud=cloud,
                        landmarks=np.empty((0, max(cloud.shape[1], 1))),
                        landmark_indices=np.empty(0, dtype=np.int64),
                        dist_matrix=np.empty((0, 0)),
                        pd_h0=np.empty((0, 2)), pd_h1=np.empty((0, 2)),
                        theta_null_h1=theta,
                    )
                    per_config[cfg.id] += time.time() - cfg_t0
                    continue

                seed = hash(f"{stock_id}{date_str}{cfg.id}{seg}{direction}") & 0xFFFFFFFF
                # 与 ph_daily_prod 同步：价格簇用更小 M + thresh 截断
                if cfg.main_signal == "DP_MICRO":
                    M = 180
                elif is_dir:
                    M = 200
                else:
                    M = 300

                tt = time.time()
                landmarks, idx, dist = maxmin_landmark(cloud, M=M, seed=seed)
                t_lm += time.time() - tt

                maxdim = 1 if seg_need_h1 else 0
                thresh = float(np.median(dist[dist > 0])) if cfg.main_signal == "DP_MICRO" else None

                tt = time.time()
                pd_h0, pd_h1 = compute_ph(dist, maxdim=maxdim, thresh=thresh)
                t_ph += time.time() - tt
                pd_count += 1

                cache[key] = PointCloudCache(
                    config_id=cfg.id, segment=seg, direction=direction,
                    raw_cloud=cloud, landmarks=landmarks,
                    landmark_indices=idx, dist_matrix=dist,
                    pd_h0=pd_h0, pd_h1=pd_h1, theta_null_h1=theta,
                )
                per_config[cfg.id] += time.time() - cfg_t0

    timings["3_pointcloud_total"] = time.time() - t_total_pd
    timings["3a_build_cloud"] = t_pc
    timings["3b_landmark"]    = t_lm
    timings["3c_ripser"]      = t_ph

    # ── 4. 因子提取（与 ph_daily_prod 同步：PD 统计量缓存避免重复计算）──
    t0 = time.time()
    stats_cache: dict = {}   # key = (config_id, segment, direction, homology)
    factor_dict = {}
    for spec in ALL_FACTOR_SPECS:
        cache_key = (spec.config_id, spec.segment, spec.direction, spec.homology)
        stats = stats_cache.get(cache_key)
        if stats is None:
            pc = cache.get((spec.config_id, spec.segment, spec.direction))
            if pc is None:
                stats = {}
            else:
                pd_arr = pc.pd_h1 if spec.homology == "H1" else pc.pd_h0
                stats = compute_pd_stats(pd_arr, pc.theta_null_h1)
            stats_cache[cache_key] = stats
        factor_dict[spec.name] = stats.get(spec.statistic, float("nan"))
    timings["4_extract_factors"] = time.time() - t0

    total = sum(v for k, v in timings.items() if not k.startswith("3a") and not k.startswith("3b") and not k.startswith("3c") and k != "3_pointcloud_total") + timings["3_pointcloud_total"]

    if verbose:
        # 耗时报告
        nan_count = sum(1 for v in factor_dict.values() if v != v)
        print("\n" + "=" * 56)
        print(f"  单股单日耗时   {stock_id}  {date_str}")
        print(f"  标定表: {'有' if calib_dir else '无(默认τ=10, 不标准化)'}")
        print("=" * 56)
        print(f"  1. 数据加载         {timings['1_load_h5']*1000:8.1f} ms")
        print(f"  2. 信号计算         {timings['2_signals']*1000:8.1f} ms")
        print(f"  3. 点云+PD 合计     {timings['3_pointcloud_total']*1000:8.1f} ms")
        print(f"       ├─ 构造点云     {timings['3a_build_cloud']*1000:8.1f} ms")
        print(f"       ├─ landmark     {timings['3b_landmark']*1000:8.1f} ms")
        print(f"       └─ ripser       {timings['3c_ripser']*1000:8.1f} ms  ← 通常是瓶颈")
        print(f"  4. 因子提取         {timings['4_extract_factors']*1000:8.1f} ms")
        print("  " + "-" * 54)
        print(f"  总计                {total*1000:8.1f} ms  ({total:.2f} s)")
        print(f"  PD 计算次数: {pd_count}   因子数: {len(factor_dict)}   NaN: {nan_count}")
        top = sorted(per_config.items(), key=lambda x: -x[1])[:5]
        print("  最耗时配置 Top5:")
        for cid, t in top:
            print(f"     {cid}: {t*1000:.1f} ms")

        # 因子健康度报告 —— 复用 inspect_output.print_report
        import sys as _sys
        _tools_dir = os.path.dirname(os.path.abspath(__file__))
        if _tools_dir not in _sys.path:
            _sys.path.insert(0, _tools_dir)
        from inspect_output import print_report
        row = {"date": date_str, "cn_code": stock_id}
        row.update(factor_dict)
        print_report(pd.DataFrame([row]),
                     source=f"{stock_id}  {date_str}")

    return total, timings, factor_dict


def _cli():
    p = argparse.ArgumentParser(description="单股单日耗时测量 + 因子健康度打印")
    p.add_argument("stock_id", nargs="?", type=str, help="如 sh600000")
    p.add_argument("date", nargs="?", type=str, help="如 20250507")
    p.add_argument("--h5", type=str, default=None, help="直接给 h5 路径")
    p.add_argument("--date", dest="date_opt", type=str, default=None)
    p.add_argument("--h5_root", type=str, default=H5_BASE)
    p.add_argument("--calib_dir", type=str, default=None)
    p.add_argument("--repeat", type=int, default=1, help="重复跑几次取均值")
    args = p.parse_args()

    if args.h5:
        file_path = args.h5
        date_str = args.date_opt or args.date
    else:
        date_str = args.date
        file_path = os.path.join(args.h5_root, date_str[:4], date_str, f"{args.stock_id}.h5")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到 h5: {file_path}")

    times = []
    for i in range(args.repeat):
        total, _, _ = time_one_stock(file_path, date_str, args.calib_dir,
                                     verbose=(i == 0))
        times.append(total)

    if args.repeat > 1:
        print(f"重复 {args.repeat} 次：均值 {np.mean(times):.2f}s  "
              f"最小 {np.min(times):.2f}s  最大 {np.max(times):.2f}s")


if __name__ == "__main__":
    _cli()