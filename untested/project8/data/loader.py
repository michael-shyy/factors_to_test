"""data/loader.py — h5 文件读取，返回 (order, tick, trade) 三个 DataFrame。"""
import os
import pandas as pd
import h5py
import hdf5plugin

os.environ['HDF5_PLUGIN_PATH'] = (
    "/home/hysheng/.local/lib/python3.10/site-packages/hdf5plugin/plugins/"
)

L2_ROOT = "/home/sharedriver/public/Level2"


def get_file_path(date: str, stock_id: str) -> str:
    """构造 h5 文件路径，date='20250102', stock_id='sh600000'"""
    year = date[:4]
    return os.path.join(L2_ROOT, year, date, f"{stock_id}.h5")


def load_daily(date: str, stock_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    读取单只股票单日 L2 数据。

    时间戳规则
    ----------
    order.OrderTime / tick.time / trade.TradeTime 均为 UNIX 秒（float64，小数部分为毫秒）。
    本函数不做任何时间过滤，原始数据原样返回，由上层决定窗口。

    Returns
    -------
    (order, tick, trade) : 三个原始 DataFrame
    """
    path = get_file_path(date, stock_id)
    with h5py.File(path, 'r') as f:
        order = pd.DataFrame(f['order/table'][:])
        tick  = pd.DataFrame(f['tick/table'][:])
        trade = pd.DataFrame(f['trade/table'][:])
    return order, tick, trade
