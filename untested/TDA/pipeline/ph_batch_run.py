"""
ph_batch_run.py
===============
TDA 因子 —— 多日批量 pipeline。
沿用 Project4 geo_batch_run.py 架构。
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
import traceback
import pickle
from datetime import datetime

import pandas as pd

from ph_daily_prod import produce_one_day, H5_BASE, DEFAULT_OUT_DIR, DEFAULT_WORKERS, DEFAULT_TIMEOUT

LOG_DIR = os.path.join(os.path.dirname(DEFAULT_OUT_DIR), "logs")


class _Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fp = open(path, "a", buffering=1)

    def log(self, msg: str = "", stdout: bool = True):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.fp.write(line + "\n")
        if stdout:
            print(line, flush=True)

    def close(self):
        self.fp.close()


def _has_h5(date_dir: str) -> bool:
    try:
        with os.scandir(date_dir) as it:
            for e in it:
                if e.name.endswith(".h5") and e.is_file():
                    return True
    except (FileNotFoundError, PermissionError):
        pass
    return False


def _scan_trading_days(h5_root: str, start: str, end: str) -> list:
    start_i, end_i = int(start), int(end)
    out = []
    for year in sorted(os.listdir(h5_root)):
        if not (year.isdigit() and len(year) == 4 and start[:4] <= year <= end[:4]):
            continue
        ydir = os.path.join(h5_root, year)
        for d in sorted(os.listdir(ydir)):
            if d.isdigit() and len(d) == 8 and start_i <= int(d) <= end_i:
                if _has_h5(os.path.join(ydir, d)):
                    out.append(d)
    return out


def _load_calibration(calib_dir: str, month: str):
    """从 configs/ 加载 tau / norm / theta parquet，返回 (tau_table, norm_table, theta_table)。"""
    def _read(name):
        p = os.path.join(calib_dir, f"{name}.parquet")
        return pd.read_parquet(p) if os.path.exists(p) else None

    tau_df   = _read("tau_calibration")
    norm_df  = _read("norm_params")
    theta_df = _read("theta_null")

    def _to_nested(df, key_cols, val_col):
        """把 parquet 转成 {stock_id: {key: val}} 嵌套 dict。"""
        if df is None:
            return {}
        sub = df[df["month"] == month] if "month" in df.columns else df
        out = {}
        for row in sub.itertuples(index=False):
            sid = row.stock_id
            if sid not in out:
                out[sid] = {}
            # key_cols 决定嵌套层级
            node = out[sid]
            for kc in key_cols[:-1]:
                k = getattr(row, kc)
                if k not in node:
                    node[k] = {}
                node = node[k]
            node[getattr(row, key_cols[-1])] = getattr(row, val_col)
        return out

    tau_table   = _to_nested(tau_df,   ["config_id", "tau_level"], "tau_ticks") if tau_df is not None else {}
    theta_table = _to_nested(theta_df, ["config_id"],              "theta")     if theta_df is not None else {}

    # norm_table: {stock_id: {signal_name: {mean, std}}}
    norm_table: dict = {}
    if norm_df is not None:
        sub = norm_df[norm_df["month"] == month] if "month" in norm_df.columns else norm_df
        for row in sub.itertuples(index=False):
            sid = row.stock_id
            if sid not in norm_table:
                norm_table[sid] = {}
            norm_table[sid][row.signal_name] = {"mean": row.mean, "std": row.std}

    return tau_table, norm_table, theta_table


def run_batch(
    start_date: str = "20200101",
    end_date:   str = "20261231",
    pkl_dir:    str = DEFAULT_OUT_DIR,
    log_dir:    str = LOG_DIR,
    h5_root:    str = H5_BASE,
    calib_dir:  str = None,
    n_workers:  int = DEFAULT_WORKERS,
    timeout:    int = DEFAULT_TIMEOUT,
    overwrite:  bool = False,
    max_days:   int = 0,
) -> dict:
    os.makedirs(pkl_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if calib_dir is None:
        calib_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    plog = _Logger(os.path.join(log_dir, f"progress_{run_id}.log"))
    elog = _Logger(os.path.join(log_dir, f"errors_{run_id}.log"))

    all_dates = _scan_trading_days(h5_root, start_date, end_date)
    todo = [d for d in all_dates
            if overwrite or not os.path.exists(os.path.join(pkl_dir, f"factors_{d}.pkl"))]
    if max_days > 0:
        todo = todo[:max_days]

    # ── 启动横幅 ────────────────────────────────────────────────────────
    bar = "═" * 70
    plog.log(bar)
    plog.log(f"  ph_batch_run  run_id={run_id}")
    plog.log(f"  区间: {start_date} ~ {end_date}    workers={n_workers}")
    plog.log(f"  交易日总数: {len(all_dates)}    已存在跳过: {len(all_dates)-len(todo)}    待跑: {len(todo)}")
    plog.log(f"  输出: {pkl_dir}")
    plog.log(f"  日志: {log_dir}/progress_{run_id}.log")
    plog.log(bar)

    if not todo:
        plog.log("没有待跑日期，退出。")
        plog.close(); elog.close()
        return {"total": len(all_dates), "ok": 0, "skip": len(all_dates), "fail": 0}

    n_ok = n_fail = 0
    fail_dates = []
    t_batch = time.time()
    prev_month = None
    tau_table = norm_table = theta_table = {}
    n_total = len(todo)

    def _fmt_eta(seconds: float) -> str:
        if seconds >= 86400:
            return f"{seconds/86400:.1f}d"
        if seconds >= 3600:
            return f"{seconds/3600:.1f}h"
        return f"{seconds/60:.1f}m"

    def _progress_bar(done: int, total: int, width: int = 30) -> str:
        frac = done / max(total, 1)
        filled = int(frac * width)
        return "▓" * filled + "░" * (width - filled)

    for i, date in enumerate(todo, 1):
        month = date[:6]
        if month != prev_month:
            t_calib = time.time()
            tau_table, norm_table, theta_table = _load_calibration(calib_dir, month)
            plog.log(f"[加载 {month} 标定表] tau={len(tau_table)}  norm={len(norm_table)}  "
                     f"theta={len(theta_table)}  耗时={time.time()-t_calib:.1f}s")
            prev_month = month

        # ── 单日开始横幅 ──────────────────────────────────────────────
        elap = time.time() - t_batch
        avg  = elap / max(i - 1, 1) if i > 1 else 0
        eta_s = avg * (n_total - i + 1) if i > 1 else 0
        pct  = (i - 1) / n_total * 100
        plog.log("")
        plog.log(f"┌─ 第 {i}/{n_total} 天: {date}  ({pct:.1f}%)  "
                 f"已用 {_fmt_eta(elap)}  预计剩 {_fmt_eta(eta_s) if i > 1 else '--'}")
        plog.log(f"│  {_progress_bar(i-1, n_total)}")

        t0 = time.time()
        try:
            produce_one_day(
                date=date,
                tau_table=tau_table,
                norm_table=norm_table,
                theta_table=theta_table,
                output_dir=pkl_dir,
                h5_root=h5_root,
                n_workers=n_workers,
                timeout=timeout,
                overwrite=overwrite,
                verbose=True,   # 显示日内股票级进度
            )
            n_ok += 1
            status = "✅"
        except Exception:
            tb = traceback.format_exc()
            elog.log(f"==== {date} ====\n{tb}", stdout=False)
            n_fail += 1
            fail_dates.append(date)
            status = "❌"

        elapsed = time.time() - t0
        # 重新算用上当前天的 ETA
        elap_now = time.time() - t_batch
        avg_now  = elap_now / i
        eta_now  = avg_now * (n_total - i)

        plog.log(f"└─ {date}  {status}  耗时 {elapsed:.1f}s    "
                 f"OK={n_ok}  失败={n_fail}    "
                 f"平均 {avg_now:.0f}s/天    ETA {_fmt_eta(eta_now)}")

    # ── 收尾 ────────────────────────────────────────────────────────────
    total_time = time.time() - t_batch
    plog.log("")
    plog.log(bar)
    plog.log(f"  完成  成功={n_ok}/{n_total}  失败={n_fail}  "
             f"总耗时={_fmt_eta(total_time)}  平均={total_time/n_total:.0f}s/天")
    if fail_dates:
        plog.log(f"  失败日期 ({len(fail_dates)} 个): {fail_dates[:10]}"
                 f"{'...' if len(fail_dates) > 10 else ''}")
        plog.log(f"  详细错误见: {log_dir}/errors_{run_id}.log")
    plog.log(bar)
    plog.close(); elog.close()

    return {"total": len(all_dates), "ok": n_ok, "fail": n_fail, "fail_dates": fail_dates}


def _cli():
    p = argparse.ArgumentParser(description="TDA 因子多日批量")
    p.add_argument("--start_date", default="20200101")
    p.add_argument("--end_date",   default="20261231")
    p.add_argument("--pkl_dir",    default=DEFAULT_OUT_DIR)
    p.add_argument("--log_dir",    default=LOG_DIR)
    p.add_argument("--h5_root",    default=H5_BASE)
    p.add_argument("--calib_dir",  default=None)
    p.add_argument("--n_workers",  type=int, default=DEFAULT_WORKERS)
    p.add_argument("--timeout",    type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--max_days",   type=int, default=0)
    args = p.parse_args()
    print(run_batch(**{k: v for k, v in vars(args).items()}))


if __name__ == "__main__":
    _cli()