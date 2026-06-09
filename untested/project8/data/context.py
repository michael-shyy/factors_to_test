"""
data/context.py — DailyDataContext，Project8 因子计算的统一输入对象。

直接扩展 Processor/dataPrepare.py 的 DailyDataContext，增加：
  - snap_features : snapshot_aligner.snapshot_features 的结果
  - tick_enriched : add_shared_features 处理后的 tick（含 probe_flag 等）
"""
import sys
import pandas as pd

sys.path.insert(0, '/home/hysheng')
from Processor.dataPrepare import DailyDataContext as _Base
from data.snapshot_aligner import snapshot_features
from data.shared_features import add_shared_features


class DailyDataContext(_Base):
    def __init__(self, tick: pd.DataFrame, trade: pd.DataFrame,
                 order: pd.DataFrame, stock_id: str = ''):
        super().__init__(tick=tick, trade=trade, order=order, stock_id=stock_id)
        self._snap_features = None
        self._tick_enriched = None

    @property
    def snap_features(self) -> pd.DataFrame:
        """四流对齐后的快照级特征，每行对应一个 tick 时刻。"""
        if self._snap_features is None:
            self._snap_features = snapshot_features(
                self._order, self._trade, self._tick
            )
        return self._snap_features

    @property
    def tick_enriched(self) -> pd.DataFrame:
        """tick + probe_flag / tick_stability / bob_symm。"""
        if self._tick_enriched is None:
            self._tick_enriched = add_shared_features(self._tick)
        return self._tick_enriched
