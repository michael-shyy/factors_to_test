"""
ph_daily_prod.py
================
TDA 因子 —— 单日生产化入口。
"""

import os
import sys

# ── sys.path：主进程和 forkserver 子进程都需要能找到 ph_factors/ ──────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HYSHENG      = "/home/hysheng"   # Processor 包的父目录

for _p in (_PROJECT_ROOT, _HYSHENG):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────
import glob
import time
import pickle
import argparse
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutTimeout
from typing import Optional

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

# ── 常量 ────────────────────────────────────────────────────────────────────

H5_BASE         = "/home/sharedriver/public/Level2"

# 项目根目录（pipeline/ 的上一级），output/ 固定放在项目内部，
# 不依赖项目叫什么名字、放在哪里。
_PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT     = os.path.join(_PROJECT_DIR, "output")
DEFAULT_OUT_DIR = os.path.join(OUTPUT_ROOT, "factors_daily_pkl")
DEFAULT_WORKERS = 20
DEFAULT_TIMEOUT = 90   # 秒/股，单股预算 ~2s（优化后），给充足余量

# ── 子进程全局状态 ───────────────────────────────────────────────────────────

_WORKER_CFG: dict = {}


def _worker_init(cfg: dict) -> None:
    """子进程初始化：重建 sys.path → numba warmup → 预导入模块。"""
    global _WORKER_CFG
    _WORKER_CFG = cfg

    # forkserver 子进程不继承父进程 sys.path，必须在这里重建
    import os, sys
    for _p in (_PROJECT_ROOT, _HYSHENG):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)

    # numba warmup + 预导入（直接按模块路径导入，不走 __init__.py）
    import traceback as _tb
    try:
        from ph_factors.landmark import maxmin_landmark
        import numpy as _np
        maxmin_landmark(_np.random.default_rng(0).random((20, 2)), M=10, seed=0)
        from ph_factors.signals import compute_all_signals          # noqa
        from ph_factors.point_cloud import build_point_cloud        # noqa
        from ph_factors.ph_kernels import compute_ph                # noqa
        from ph_factors.pd_statistics import compute_pd_stats       # noqa
        from ph_factors.registry import CONFIGS, ALL_FACTOR_SPECS   # noqa
    except Exception as e:
        print(f"[WORKER INIT WARN] {e}", flush=True)
        print(f"[WORKER INIT TRACEBACK]\n{_tb.format_exc()}", flush=True)
        print(f"[WORKER INIT SYSPATH] {sys.path[:6]}", flush=True)
        print(f"[WORKER INIT PROJECT_ROOT] {_PROJECT_ROOT}", flush=True)


def _worker_process(file_path: str) -> dict:
    """子进程入口：单股计算。必须是模块级函数才能被 forkserver pickle。"""
    return _process_one_stock(file_path, **_WORKER_CFG)


# ── 数据加载 ─────────────────────────────────────────────────────────────────

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


# ── 单股计算 ─────────────────────────────────────────────────────────────────

def _process_one_stock(
    file_path: str,
    tau_table: dict,       # {stock_id: {config_id: {tau_level: int}}}
    norm_table: dict,      # {stock_id: {signal_name: {mean, std}}}
    theta_table: dict,     # {stock_id: {config_id: float}}
    date_str: str,
) -> dict:
    """
    单股单日全量因子计算。

    Returns dict:
      'stock_id': str
      'factors':  dict[factor_name, float]  （536 个）
    或出错时:
      {'__error__': True, '__tb__': str, 'stock_id': str}
    """
    stock_id = os.path.basename(file_path).replace(".h5", "")

    try:
        order_df, tick_df, trade_df = _load_h5(file_path)
        if len(tick_df) < 10:
            return {"stock_id": stock_id, "factors": {}}

        from ph_factors.signals import compute_all_signals
        from ph_factors.point_cloud import build_point_cloud
        from ph_factors.landmark import maxmin_landmark
        from ph_factors.ph_kernels import compute_ph
        from ph_factors.pd_statistics import compute_pd_stats
        from ph_factors.registry import CONFIGS, ALL_FACTOR_SPECS, PointCloudCache

        norm_params  = norm_table.get(stock_id, {})
        tau_lookup   = tau_table.get(stock_id, {})
        theta_lookup = theta_table.get(stock_id, {})

        signals = compute_all_signals(
            order_df, tick_df, date_str,
            trade_df=trade_df, stock_id=stock_id,
            norm_params=norm_params or None,
        )

        # 构造所有 (config, segment, direction) 的 PointCloudCache
        #
        # 性能优化要点：
        #  1. landmark M=300（实测 ripser 比 M=500 快 ~3x，环检测精度几乎无损）
        #  2. 某 segment 只需 H0 时只算 maxdim=0（比 H0+H1 快 ~3x）
        #  3. 方向化点云(BUY/SELL)只服务 H1 MAXPERS，用更激进的 M=200
        cache: dict = {}
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
                    tau = cfg_tau.get(cfg.tau_level, 10)  # 默认 10 桶 = 30 秒

                    is_dir = direction in ("BUY", "SELL")
                    # 方向化点云只出 H1 MAXPERS；非方向化才出全套
                    seg_need_h1 = need_h1 or is_dir
                    seg_need_h0 = (seg in h0_segs) and (not is_dir)
                    if not (seg_need_h1 or seg_need_h0):
                        continue

                    cloud = build_point_cloud(cfg, seg, direction, signals, tau)
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
                        continue
                    seed = hash(f"{stock_id}{date_str}{cfg.id}{seg}{direction}") & 0xFFFFFFFF
                    # 价格簇(DP_MICRO)点云高度退化(大量 Δp≈0),ripser 在小尺度
                    # 上爆炸。专门给它们更小 M + ripser thresh 截断超大尺度无意义环,
                    # 实测能砍掉 70%+ ripser 时间且不影响金融含义。
                    if cfg.main_signal == "DP_MICRO":
                        M = 180
                    elif is_dir:
                        M = 200
                    else:
                        M = 300
                    landmarks, idx, dist = maxmin_landmark(cloud, M=M, seed=seed)
                    maxdim = 1 if seg_need_h1 else 0
                    # 价格簇加 thresh: 距离矩阵的中位数,只看小到中尺度的拓扑结构
                    thresh = float(np.median(dist[dist > 0])) if cfg.main_signal == "DP_MICRO" else None
                    pd_h0, pd_h1 = compute_ph(dist, maxdim=maxdim, thresh=thresh)
                    cache[key] = PointCloudCache(
                        config_id=cfg.id, segment=seg, direction=direction,
                        raw_cloud=cloud, landmarks=landmarks,
                        landmark_indices=idx, dist_matrix=dist,
                        pd_h0=pd_h0, pd_h1=pd_h1,
                        theta_null_h1=theta,
                    )

        # 提取因子值
        # 关键优化：每份 PD 的全套统计量只算一次，缓存后所有 spec 直接查。
        # 560 个 spec 共享 ~40 个 PD，避免对同一 PD 重复算 13 次。
        stats_cache: dict = {}   # key = (config_id, segment, direction, homology)
        factor_dict: dict = {}
        for spec in ALL_FACTOR_SPECS:
            cache_key = (spec.config_id, spec.segment, spec.direction, spec.homology)
            stats = stats_cache.get(cache_key)
            if stats is None:
                pc = cache.get((spec.config_id, spec.segment, spec.direction))
                if pc is None:
                    stats = {}   # 标记“无 PD”
                else:
                    pd_arr = pc.pd_h1 if spec.homology == "H1" else pc.pd_h0
                    stats = compute_pd_stats(pd_arr, pc.theta_null_h1)
                stats_cache[cache_key] = stats
            factor_dict[spec.name] = stats.get(spec.statistic, float("nan"))

        return {"stock_id": stock_id, "factors": factor_dict}

    except Exception:
        return {
            "__error__": True,
            "__tb__": traceback.format_exc(),
            "stock_id": stock_id,
        }


# ── 单日生产入口 ──────────────────────────────────────────────────────────────

def produce_one_day(
    date:        str,
    tau_table:   dict,
    norm_table:  dict,
    theta_table: dict,
    output_dir:  str  = DEFAULT_OUT_DIR,
    h5_root:     str  = H5_BASE,
    n_workers:   int  = DEFAULT_WORKERS,
    timeout:     int  = DEFAULT_TIMEOUT,
    mp_method:   str  = "forkserver",
    overwrite:   bool = False,
    verbose:     bool = True,
    debug_serial: bool = False,
    max_files:   int  = 0,
) -> str:
    """
    单日生产化：扫描 H5 → 多进程跑因子 → 保存 pkl。

    Returns: pkl 文件绝对路径。
    """
    date_clean = date.replace("-", "")
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f"factors_{date_clean}.pkl")

    if os.path.exists(pkl_path) and not overwrite:
        if verbose:
            print(f"[SKIP] {pkl_path} 已存在")
        return pkl_path

    date_dir = os.path.join(h5_root, date_clean[:4], date_clean)
    if not os.path.isdir(date_dir):
        raise FileNotFoundError(f"数据目录不存在: {date_dir}")

    files = sorted(glob.glob(os.path.join(date_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"{date_dir} 下没有 .h5 文件")
    if max_files > 0:
        files = files[:max_files]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ph_daily_prod  日期={date_clean}  workers={n_workers}"
              f"{'  [SERIAL]' if debug_serial else ''}")
        print(f"  股票数: {len(files)}")
        print(f"  输出  : {pkl_path}")
        print(f"{'='*60}")

    cfg = dict(
        tau_table=tau_table,
        norm_table=norm_table,
        theta_table=theta_table,
        date_str=date_clean,
    )

    rows   = []
    errors = []
    t0     = time.time()

    if debug_serial:
        _worker_init(cfg)
        for f in files:
            result = _process_one_stock(f, **cfg)
            if result.get("__error__"):
                errors.append(result)
            elif result.get("factors"):
                rows.append(result)
    else:
        mp_ctx = multiprocessing.get_context(mp_method)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
            initializer=_worker_init,
            initargs=(cfg,),
        ) as executor:
            futures = {executor.submit(_worker_process, f): f for f in files}
            n_total = len(files)
            n_done  = 0
            t_prog  = time.time()
            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    result = fut.result(timeout=timeout)
                except FutTimeout:
                    errors.append({"stock_id": os.path.basename(f).replace(".h5", ""),
                                    "error": "TIMEOUT"})
                except Exception as e:
                    errors.append({"stock_id": os.path.basename(f).replace(".h5", ""),
                                    "error": str(e), "__tb__": traceback.format_exc()})
                else:
                    if result.get("__error__"):
                        errors.append(result)
                    elif result.get("factors"):
                        rows.append(result)
                    del result
                n_done += 1
                # 每完成 100 只 或 每 15 秒 打一次进度
                if verbose and (n_done % 100 == 0
                                or time.time() - t_prog > 15
                                or n_done == n_total):
                    elap  = time.time() - t0
                    rate  = n_done / max(elap, 0.001)
                    eta_s = (n_total - n_done) / max(rate, 0.001)
                    sys.stdout.write(
                        f"  [{n_done:>5}/{n_total}] OK={len(rows)} err={len(errors)}  "
                        f"{rate:.1f}股/s  ETA {eta_s/60:.1f}m\n"
                    )
                    sys.stdout.flush()
                    t_prog = time.time()

    elapsed = time.time() - t0
    if verbose:
        print(f"[STAT] OK={len(rows)}  error={len(errors)}  "
              f"total={len(files)}  耗时={elapsed:.1f}s")
        if errors:
            for e in errors[:3]:
                print(f"  - {e.get('stock_id')}  {e.get('error', e.get('__tb__', ''))[:120]}")

    if not rows:
        raise RuntimeError(f"{date_clean} 全部股票失败，无数据输出")

    # 拼宽表
    out_df = pd.DataFrame([
        {"cn_code": r["stock_id"], **r["factors"]} for r in rows
    ])
    out_df.insert(0, "date", int(date_clean))
    out_df = out_df.sort_values("cn_code").reset_index(drop=True)

    with open(pkl_path, "wb") as fp:
        pickle.dump(out_df, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"[OK] {pkl_path}  shape={out_df.shape}")

    return pkl_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(description="TDA 因子单日生产")
    p.add_argument("date", type=str, help="日期，如 20250507")
    p.add_argument("--output_dir",   type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--h5_root",      type=str, default=H5_BASE)
    p.add_argument("--n_workers",    type=int, default=DEFAULT_WORKERS)
    p.add_argument("--timeout",      type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--mp_method",    type=str, default="forkserver",
                   choices=["forkserver", "spawn", "fork"])
    p.add_argument("--overwrite",    action="store_true")
    p.add_argument("--debug_serial", action="store_true")
    p.add_argument("--max_files",    type=int, default=0)
    args = p.parse_args()

    # 实际生产时 tau/norm/theta 从 parquet 加载；CLI 模式用空 dict（无标准化）
    produce_one_day(
        date         = args.date,
        tau_table    = {},
        norm_table   = {},
        theta_table  = {},
        output_dir   = args.output_dir,
        h5_root      = args.h5_root,
        n_workers    = args.n_workers,
        timeout      = args.timeout,
        mp_method    = args.mp_method,
        overwrite    = args.overwrite,
        debug_serial = args.debug_serial,
        max_files    = args.max_files,
    )


if __name__ == "__main__":
    _cli()