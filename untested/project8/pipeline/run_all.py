"""
pipeline/run_all.py — 全量历史因子计算入口

用法
----
python3 pipeline/run_all.py                        # 全部可用年份
python3 pipeline/run_all.py --years 2025
python3 pipeline/run_all.py --start 20240102 --end 20260630
python3 pipeline/run_all.py --workers 40
python3 pipeline/run_all.py --no-resume            # 重新计算（忽略缓存）

输出
----
output/factor_panels/{date}.parquet     每日截面（断点续算）
output/factor_panels/{factor}.parquet   每因子面板（Nova 兼容）
output/run_summary.csv                  运行摘要
"""

import argparse
import glob
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/hysheng")

from pipeline.daily_runner import run_one

L2_ROOT = "/home/sharedriver/public/Level2"
OUT_DIR = "/home/hysheng/project8/output/factor_panels"


# ── numba 预热（在每个 worker 进程启动时执行一次）───────────────────────────

def _warmup():
    """触发 numba JIT 编译，让进程后续调用直接用缓存。"""
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, sys
    sys.path.insert(0, "/home/hysheng/project8")
    sys.path.insert(0, "/home/hysheng")
    # 只要 import 并调用一次这两个 njit 函数就能触发编译
    from features.lz_complexity.lz76 import _rolling_lz76_njit
    from features.mutual_info.ksg_estimator import _ksg_mi_core
    dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5, dtype=np.int8)
    _rolling_lz76_njit(dummy, 10, 1.0, 1)
    x = np.random.randn(20); y = np.random.randn(20)
    _ksg_mi_core(x, y, 3)


# ── worker ────────────────────────────────────────────────────────────────

def _worker(args):
    date, stock_id = args
    return run_one(date, stock_id)


# ── 单日截面（复用传入的 executor，不自己创建）────────────────────────────

def _run_date_with_pool(
    date: str,
    executor: ProcessPoolExecutor,
    resume: bool,
) -> pd.DataFrame | None:
    cache = os.path.join(OUT_DIR, f"{date}.parquet")
    if resume and os.path.exists(cache):
        return pd.read_parquet(cache)

    files = glob.glob(os.path.join(L2_ROOT, date[:4], date, "*.h5"))
    if not files:
        return None

    tasks = [(date, os.path.basename(f).replace(".h5", "")) for f in files]
    rows = []
    futs = {executor.submit(_worker, t): t for t in tasks}
    for fut in as_completed(futs):
        r = fut.result()
        if r:
            rows.append(r)

    if not rows:
        return None

    df = pd.DataFrame(rows).set_index(["date", "stock_id"]).sort_index()
    df = df.dropna(axis=1, how="all")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(cache)
    return df


# ── 主流程 ────────────────────────────────────────────────────────────────

def run_all(
    years: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    n_workers: int = 40,
    resume: bool = True,
) -> pd.DataFrame:
    # 收集所有日期
    all_dates = []
    scan_years = years or [y for y in os.listdir(L2_ROOT) if y.isdigit()]
    for yr in sorted(scan_years):
        yr_dir = os.path.join(L2_ROOT, yr)
        if not os.path.isdir(yr_dir):
            continue
        for d in sorted(os.listdir(yr_dir)):
            if d.isdigit() and len(d) == 8:
                all_dates.append(d)
    if start:
        all_dates = [d for d in all_dates if d >= start]
    if end:
        all_dates = [d for d in all_dates if d <= end]

    print(f"共 {len(all_dates)} 个交易日，{n_workers} 核，resume={resume}")
    print(f"日期范围: {all_dates[0]} ~ {all_dates[-1]}")
    print(f"输出目录: {OUT_DIR}\n")

    summary_rows = []
    panels = []
    t_total = time.time()

    # 整个运行期间只创建一次进程池，initializer 预热 numba
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_warmup) as executor:
        for date in tqdm(all_dates, desc="总进度"):
            t0 = time.time()
            df = _run_date_with_pool(date, executor, resume)
            elapsed = time.time() - t0

            if df is None or df.empty:
                continue

            panels.append(df)
            nan_pct = round(df.isna().mean().mean() * 100, 1)
            summary_rows.append({
                "date": date, "n_stocks": len(df),
                "n_factors": len(df.columns),
                "elapsed_s": round(elapsed, 1), "nan_pct": nan_pct,
            })
            print(f"  {date}: {len(df)}股 × {len(df.columns)}因子  {elapsed:.0f}s  nan={nan_pct:.1f}%")

    print(f"\n全量完成，总耗时 {(time.time()-t_total)/60:.1f}min")

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels).sort_index()
    _export_per_factor(panel)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(os.path.dirname(OUT_DIR), "run_summary.csv"), index=False)
    print(summary_df.to_string(index=False))
    return panel


def _export_per_factor(panel: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for col in tqdm(panel.columns, desc="导出因子"):
        df = panel[col].unstack("stock_id").sort_index()
        df.index = df.index.astype(int)
        df.to_parquet(os.path.join(OUT_DIR, f"{col}.parquet"))
    print(f"已导出 {len(panel.columns)} 个因子")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--years",   nargs="+")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all(years=args.years, start=args.start, end=args.end,
            n_workers=args.workers, resume=args.resume)

