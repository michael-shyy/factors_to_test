"""
data/preprocessor.py — 在 Processor/ 已有逻辑之上的薄封装，补充量级分档。

直接复用：
  DataProcessor.filter_time_window
  DataProcessor.long_short_tagging
  DataProcessor.build_order_lifecycle
  DataProcessor.extract_depth_cancel_rates
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/hysheng')
from Processor.processor import DataProcessor


def filter_continuous_auction(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """截取 09:30–14:00 连续竞价时段。"""
    return DataProcessor.filter_time_window(df, time_col)


def tag_orders(
    order: pd.DataFrame,
    trade: pd.DataFrame,
    short_pct: float = 0.10,
    long_pct: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """打长单/短单/大单标签，返回 (tagged_order, tagged_trade)。"""
    return DataProcessor.long_short_tagging(order, trade,
                                            short_pct=short_pct,
                                            long_pct=long_pct)


def build_lifecycle(
    order: pd.DataFrame,
    trade: pd.DataFrame,
    stock_id: str = '',
) -> pd.DataFrame:
    """重建委托生命周期，附 filled_vol/filled_ratio/cancel_time/order_type_tag。"""
    return DataProcessor.build_order_lifecycle(order, trade, stock_id=stock_id)


def tag_size_bucket(
    order: pd.DataFrame,
    n_buckets: int = 5,
    amt_col: str = '_order_amt',
) -> pd.DataFrame:
    """
    在 order 上新增 size_bucket 列（1=最小 … n_buckets=最大），按委托金额等频分档。

    时间戳规则：只依赖 OrderPrice/OrderVolume，无时间前视风险。
    """
    df = order.copy()
    df[amt_col] = df['OrderPrice'] * df['OrderVolume']
    df['size_bucket'] = pd.qcut(
        df[amt_col], q=n_buckets, labels=False, duplicates='drop'
    ).astype('Int8') + 1
    df = df.drop(columns=[amt_col])
    return df


def depth_cancel_rates(
    order: pd.DataFrame,
    trade: pd.DataFrame,
    tick: pd.DataFrame,
    n_levels: int = 5,
) -> pd.DataFrame:
    """各档撤单率（bid_cancel_rate_1~5, ask_cancel_rate_1~5），index=tick时间。"""
    return DataProcessor.extract_depth_cancel_rates(order, trade, tick, n_levels=n_levels)
