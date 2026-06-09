"""
consolidate_panel.py
====================
把 factors_daily_pkl/ 下的每日 pkl 合并成因子级 parquet 宽表。
输出：configs/../output/factors_panel/{factor_name}.parquet，date × stock_id。
"""

# ── sys.path bootstrap ──────────────────────────────────────────────────────
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa
# ───────────────────────────────────────────────────────────────────────────

import os
import glob
import pickle
import argparse

import pandas as pd
from tqdm.auto import tqdm

from ph_daily_prod import DEFAULT_OUT_DIR

DEFAULT_PANEL_DIR = os.path.join(os.path.dirname(DEFAULT_OUT_DIR), "factors_panel")


def consolidate(
    pkl_dir:   str = DEFAULT_OUT_DIR,
    panel_dir: str = DEFAULT_PANEL_DIR,
    overwrite: bool = False,
) -> None:
    os.makedirs(panel_dir, exist_ok=True)
    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, "factors_*.pkl")))
    if not pkl_files:
        print(f"[WARN] {pkl_dir} 下没有 pkl 文件")
        return

    print(f"读取 {len(pkl_files)} 个 pkl ...")
    frames = []
    for f in tqdm(pkl_files):
        with open(f, "rb") as fp:
            df = pickle.load(fp)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    factor_cols = [c for c in all_df.columns if c not in ("date", "cn_code")]

    print(f"共 {len(factor_cols)} 个因子，{all_df['date'].nunique()} 天，"
          f"{all_df['cn_code'].nunique()} 只股票")
    print(f"写出到 {panel_dir} ...")

    for col in tqdm(factor_cols):
        out_path = os.path.join(panel_dir, f"{col}.parquet")
        if os.path.exists(out_path) and not overwrite:
            continue
        pivot = all_df.pivot(index="date", columns="cn_code", values=col)
        pivot.to_parquet(out_path)

    print("[OK] 面板汇总完成")


def _cli():
    p = argparse.ArgumentParser(description="TDA 因子面板汇总")
    p.add_argument("--pkl_dir",   default=DEFAULT_OUT_DIR)
    p.add_argument("--panel_dir", default=DEFAULT_PANEL_DIR)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    consolidate(args.pkl_dir, args.panel_dir, args.overwrite)


if __name__ == "__main__":
    _cli()
