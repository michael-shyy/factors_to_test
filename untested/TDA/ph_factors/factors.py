"""从 CloudConfig 列表展开全量 FactorSpec（536 个）。"""
from __future__ import annotations
from typing import List
from ph_factors.registry import CloudConfig, FactorSpec

STATS = [
    "MAXPERS", "TOP3_MEAN", "L2", "MAXPERS_RATIO",
    "ENT", "P75_MINUS_P25", "SKEW",
    "COUNT_LOW", "COUNT_HIGH", "MAXGAP",
    "MEAN_BIRTH", "MEAN_DEATH", "LIFETIME_OVER_BIRTH_MEAN",
]


def enumerate_all_factors(configs: List[CloudConfig]) -> List[FactorSpec]:
    """展开全 536 个 FactorSpec。"""
    specs: List[FactorSpec] = []
    for cfg in configs:
        # 主因子：13 统计量 × H0/H1 × 时段
        for homology, segs in [("H1", cfg.h1_segments), ("H0", cfg.h0_segments)]:
            for stat in STATS:
                for seg in segs:
                    specs.append(FactorSpec(
                        name=f"GEO_PH_{cfg.id}_{homology}_{stat}_{seg}",
                        config_id=cfg.id,
                        homology=homology,
                        statistic=stat,
                        segment=seg,
                        direction=None,
                    ))
        # 方向化 MAXPERS：仅 H1 FULL，仅有 BUY/SELL 的配置
        if any(d in ("BUY", "SELL") for d in cfg.directions):
            for direction in ("BUY", "SELL"):
                specs.append(FactorSpec(
                    name=f"GEO_PH_{cfg.id}_H1_MAXPERS_{direction}_FULL",
                    config_id=cfg.id,
                    homology="H1",
                    statistic="MAXPERS",
                    segment="FULL",
                    direction=direction,
                ))
    return specs
