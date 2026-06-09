"""因子注册：加载配置 YAML，生成全量 FactorSpec 列表。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml

# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class CloudConfig:
    id: str
    cluster: str
    paradigm: str                   # "B" | "C"
    main_signal: str
    tau_level: str                  # "short" | "mid" | "long"
    env_dims: List[str]
    total_dim: int
    embed_order: int
    h1_segments: List[str]
    h0_segments: List[str]
    directions: List[Optional[str]] # [None] | [None, "BUY", "SELL"]


@dataclass
class PointCloudCache:
    config_id: str
    segment: str                    # FULL | OPEN
    direction: Optional[str]        # None | BUY | SELL
    raw_cloud: "np.ndarray"         # (N, d)
    landmarks: "np.ndarray"         # (M, d)
    landmark_indices: "np.ndarray"  # (M,)
    dist_matrix: "np.ndarray"       # (M, M)
    pd_h0: "np.ndarray"             # (k0, 2)
    pd_h1: "np.ndarray"             # (k1, 2)
    theta_null_h1: float
    meta: dict = field(default_factory=dict)


@dataclass
class FactorSpec:
    name: str
    config_id: str
    homology: str                   # H0 | H1
    statistic: str
    segment: str                    # FULL | OPEN
    direction: Optional[str]        # None | BUY | SELL


# ── 加载配置 ──────────────────────────────────────────────────────────────────

_YAML_PATH = Path(__file__).parent.parent / "configs" / "pointcloud_configs.yaml"


def _load_configs(path: Path = _YAML_PATH) -> List[CloudConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    configs = []
    for c in raw["configs"]:
        # YAML null → Python None
        directions = [None if d is None else d for d in c.get("directions", [None])]
        configs.append(CloudConfig(
            id=c["id"],
            cluster=c["cluster"],
            paradigm=c["paradigm"],
            main_signal=c["main_signal"],
            tau_level=c["tau_level"],
            env_dims=c.get("env_dims", []),
            total_dim=c["total_dim"],
            embed_order=c.get("embed_order", 2),
            h1_segments=c["h1_segments"],
            h0_segments=c["h0_segments"],
            directions=directions,
        ))
    return configs


CONFIGS: List[CloudConfig] = _load_configs()

# 延迟导入避免循环
def _build_specs():
    from ph_factors.factors import enumerate_all_factors
    return enumerate_all_factors(CONFIGS)

ALL_FACTOR_SPECS: List[FactorSpec] = _build_specs()
