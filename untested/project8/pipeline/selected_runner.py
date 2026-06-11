"""pipeline/selected_runner.py — 计算单日 6 个入库因子，输出 pkl。

回测区间 20240102–20260424，入库因子：
    LZ_diff, LZ_syntax_kl, LZ_multi, LZ_highvol,
    BURST_TYPE_PROG_sell, BURST_TYPE_SHOCK_buy

用法
----
    python -m pipeline.selected_runner --date 20250102 [--out ./output] [--workers 20]

环境变量
--------
    L2_ROOT          L2 数据根目录，默认 /home/sharedriver/public/Level2
    HDF5_PLUGIN_PATH hdf5plugin 插件目录（非默认安装路径时需设置）

输出
----
    {out}/{date}_selected.pkl   — DataFrame，index=(date, stock_id)，6 列因子
"""
import argparse
import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# 确保项目根在 sys.path（直接运行或作为模块均可）
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_daily, L2_ROOT
from data.context import DailyDataContext
from features.lz_complexity.factor_builder import build_lz_factors
from features.burst_silence.factor_builder import build_burst_silence_factors

_SELECTED = frozenset({
    'LZ_diff', 'LZ_syntax_kl', 'LZ_multi', 'LZ_highvol',
    'BURST_TYPE_PROG_sell', 'BURST_TYPE_SHOCK_buy',
})


def _run_one(args):
    date, stock_id = args
    try:
        order, tick, trade = load_daily(date, stock_id)
        ctx = DailyDataContext(tick=tick, trade=trade, order=order, stock_id=stock_id)
        row = {'date': date, 'stock_id': stock_id}
        for builder in (build_lz_factors, build_burst_silence_factors):
            row.update({k: v for k, v in builder(ctx).items() if k in _SELECTED})
        return row
    except Exception as e:
        print(f'[WARN] {date}/{stock_id}: {e}')
        return None


def run_day(date: str, out_dir: str = './output', n_workers: int = 20) -> pd.DataFrame:
    """计算单日截面，保存为 pkl，返回 DataFrame。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{date}_selected.pkl')

    files = glob.glob(os.path.join(L2_ROOT, date[:4], date, '*.h5'))
    if not files:
        raise FileNotFoundError(f'No h5 files found for {date} under {L2_ROOT}')

    tasks = [(date, os.path.basename(f).replace('.h5', '')) for f in files]

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                rows.append(r)

    df = pd.DataFrame(rows).set_index(['date', 'stock_id']).sort_index()

    with open(out_path, 'wb') as f:
        pickle.dump(df, f)
    print(f'Saved {len(df)} stocks → {out_path}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date',    required=True, help='交易日，格式 YYYYMMDD')
    parser.add_argument('--out',     default='./output', help='输出目录')
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    run_day(args.date, args.out, args.workers)
