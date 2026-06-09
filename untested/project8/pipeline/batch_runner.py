"""pipeline/batch_runner.py — 多进程批量计算，输出兼容 Nova 的 parquet 格式。"""
import os
import glob
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from pipeline.daily_runner import run_one

L2_ROOT   = '/home/sharedriver/public/Level2'
OUT_DIR   = '/home/hysheng/project8/output/factor_panels'


def _worker(args):
    date, stock_id = args
    return run_one(date, stock_id)


def build_daily_panel(
    date: str,
    n_workers: int = 20,
    save: bool = True,
) -> pd.DataFrame:
    """
    计算单日截面因子，多进程并行。

    Returns
    -------
    DataFrame，index=(date, stock_id)，列为各因子。
    已存 parquet 则直接读取（断点续算）。
    """
    cache = os.path.join(OUT_DIR, f'{date}.parquet')
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    year = date[:4]
    files = glob.glob(os.path.join(L2_ROOT, year, date, '*.h5'))
    if not files:
        return pd.DataFrame()

    tasks = [(date, os.path.basename(f).replace('.h5', '')) for f in files]

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=date):
            r = fut.result()
            if r:
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index(['date', 'stock_id']).sort_index()
    df = df.dropna(axis=1, how='all')

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        df.to_parquet(cache)

    return df


def build_panel(
    dates: list[str] | None = None,
    year: str = '2025',
    n_workers: int = 20,
) -> pd.DataFrame:
    """
    批量处理多个交易日，拼接后存为独立 parquet（每个因子一个文件，兼容 factor_io）。
    """
    if dates is None:
        dirs = sorted(glob.glob(os.path.join(L2_ROOT, year, '*/')))
        dates = [os.path.basename(d.rstrip('/')) for d in dirs]

    panels = []
    for date in tqdm(dates, desc='总进度'):
        df = build_daily_panel(date, n_workers=n_workers)
        if not df.empty:
            panels.append(df)

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels).sort_index()
    _export_per_factor(panel)
    return panel


def _export_per_factor(panel: pd.DataFrame) -> None:
    """
    将面板拆分为每个因子一个 parquet 文件。
    index = int YYYYMMDD，columns = 股票代码字符串。
    与 Backtest/factor_io.read_factor 完全兼容。
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    for col in panel.columns:
        df = panel[col].unstack(level='stock_id').sort_index()
        df.index = df.index.astype(int)
        df.to_parquet(os.path.join(OUT_DIR, f'{col}.parquet'))
