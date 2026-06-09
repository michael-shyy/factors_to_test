# TDA — 因子生产规范

> **用途**:本文档是 Claude Code 实现 TDA 因子库的唯一权威规范。
> 所有因子定义、命名规则、依赖关系、阶段划分都以本文档为准。
> 如和其他文档冲突,以本文档为准。
>
> **版本**:v1.0 (production-ready)
> **作者**:Michael / hysheng
> **目标**:在 A 股 5000 只股票 × 6 年区间生产约 520 个 TDA 因子。

---

## 0. TL;DR(给 Claude Code 看的速读)

- 生产 **520 个 TDA 因子**,基于 14 个点云配置 × PD 统计量 × 时段切片
- 核心技术栈:`ripser` 算持续同调 + `maxmin landmark` 子采样 + 沿用 Project4 多进程框架
- **三阶段生产**:配置粗扫 → 因子展开 → 全样本生产
- 输出格式:parquet 面板,完全对齐 Project3/4
- 单入口:`generate_factors(date: str) -> pd.DataFrame`

---

## 1. 项目背景与论证

### 1.1 项目定位

TDA 项目在 Project4 几何因子库基础上,引入拓扑数据分析(TDA)。

**核心假设**:订单流延迟嵌入点云中的 H1 环结构对应市场中的"挂-吃-撤-补"等周期性循环行为。这种闭合性质是 Project4 局部几何因子(密度、维数、子空间、各向异性)在原理上看不见的——所有现有几何因子都是局部统计,无法感知"轨迹是不是绕回了原点"。

### 1.2 与 Project4 的关系

| 维度 | Project4 | TDA |
|---|---|---|
| 数学语言 | 微分几何 / 线性代数 | 代数拓扑 / 持续同调 |
| 度量对象 | 局部形状(密度、维数、各向异性) | 全局拓扑(连通团、闭合环) |
| 对距离尺度 | 敏感 | 鲁棒(只看相对顺序) |
| 主因子定位 | GEO_DENS, GEO_SUBSPACE, GEO_GRASS | GEO_PH_H1_MAXPERS |
| 预期正交性 | — | 高(数学结构根本不重叠) |

### 1.3 数据基础

完整 L2 数据可用,无降级:

- 逐笔委托流(order-by-order):提供撤单方向、挂单存活时间
- 定时快照(snapshot):提供盘口五档量价
- 成交明细:提供成交方向、成交强度

---

## 2. 核心概念定义

### 2.1 延迟嵌入

对信号 `s(t)`,延迟嵌入向量为:

```
x_t = [s_t, s_{t-τ}]              # 2D 延迟(默认)
x_t = [s_t, s_{t-τ}, s_{t-2τ}]    # 3D 延迟
```

τ 是延迟参数,必须扫描多个值。

### 2.2 范式

| 范式 | 形式 | 用途 |
|---|---|---|
| B | `[s_t, s_{t-τ}]` | 纯延迟嵌入,主攻 H1 环 |
| C | `[s_t, s_{t-τ}, e^(1)_t, ...]` | 延迟 + 环境维,主攻 H1+H0 |

**范式 A(多变量同时刻)在本项目中不使用**——它的 H0 因子和 Project4 几何因子语义重叠,且偏离"循环检测"核心论证。

### 2.3 Persistence Diagram (PD)

PD 是 `(birth, death)` 对的集合,记 `pd_h0` 和 `pd_h1` 分别为 H0、H1 的持久性对。
持久性 `persistence = death - birth`。

### 2.4 Landmark 子采样

日内点云通常 5000-20000 点,直接算 ripser 性能爆炸。
使用 **maxmin landmark** 选 400-600 个代表点,witness complex 近似全点云 PH。

### 2.5 持久性阈值 θ_null

通过随机排列(时间打乱)信号后构造点云,计算 100 次的 H1 持久性 95% 分位数,作为"无循环零假设"下的噪声水平。
**θ_null 每月按股票分组重新标定**。

---

## 3. 点云配置(14 个,完整清单)

### 3.1 配置参数规范

每个配置由以下字段唯一确定:

```yaml
config_id: str            # 如 "C01"
paradigm: Enum["B", "C"]  # 范式
main_signal: str          # 延迟主信号名,见 §3.3
tau: Enum["short", "mid", "long"]  # τ 档位,实际值在 §3.4 标定
env_dims: list[str]       # 环境维列表(范式 B 为空)
total_dim: int            # 总维度
embed_order: int          # 延迟阶数,默认 2
```

### 3.2 配置清单

| ID | 簇 | 范式 | 主信号 | τ | 环境维 | 总维 | H1 时段 | H0 时段 | 因子数 |
|---|---|---|---|---|---|---|---|---|---|
| C01 | OFI | B | OFI_NET | short | — | 2 | FULL, OPEN | FULL, OPEN | 52 |
| C02 | OFI | B | OFI_NET | mid | — | 2 | FULL, OPEN | FULL, OPEN | 52 |
| C03 | OFI | B | OFI_NET | long | — | 2 | FULL | FULL | 26 |
| C04 | OFI | C | OFI_NET | mid | [SPREAD] | 3 | FULL | FULL | 26 |
| C05 | OFI | C | OFI_NET | mid | [DEPTH5] | 3 | FULL, OPEN | FULL, OPEN | 52 |
| C07 | IMBAL | B | OB_IMBAL | short | — | 2 | FULL, OPEN | FULL, OPEN | 52 |
| C08 | IMBAL | B | OB_IMBAL | mid | — | 2 | FULL, OPEN | FULL, OPEN | 52 |
| C09 | IMBAL | C | OB_IMBAL | mid | [SPREAD] | 3 | FULL | FULL | 26 |
| C11 | PRICE | B | DP_MICRO | short | — | 2 | FULL | FULL | 26 |
| C12 | PRICE | B | DP_MICRO | mid | — | 2 | FULL | FULL | 26 |
| C13 | PRICE | C | DP_MICRO | mid | [RV_PROXY] | 3 | FULL | FULL | 26 |
| C14 | PRICE | C | DP_MICRO | mid | [SPREAD, DEPTH5] | 4 | FULL | FULL | 26 |
| C15 | CANCEL | B | CANCEL_NET | mid | — | 2 | FULL | FULL | 26 |
| C16 | CANCEL | C | CANCEL_NET | mid | [CANCEL_RATE] | 3 | FULL, OPEN | FULL, OPEN | 52 |
| C17 | CANCEL | C | CANCEL_NET | mid | [LIFE_TIME] | 3 | FULL | FULL | 26 |

**合计:520 个因子**。

注:跳过 C06、C10、C18、C19、C20 是因为前期方案中砍掉的配置,保留 ID gap 以便未来扩展。

### 3.3 主信号定义

| 信号名 | 完整定义 | 数据源 | 取值范围 |
|---|---|---|---|
| `OFI_NET` | `OFI` 净额(全档,5档加权),按 Cont et al. (2014) 标准定义 | 逐笔委托 + 快照 | 围绕 0 |
| `OB_IMBAL` | 盘口失衡 `(V_bid - V_ask) / (V_bid + V_ask)`,5档总量 | 快照 | [-1, 1] |
| `DP_MICRO` | 微观中间价变动 `mid_t - mid_{t-1}` | 快照 | 围绕 0 |
| `CANCEL_NET` | 撤单方向净额 `cancel_buy - cancel_sell`,逐笔计 | 逐笔委托 | 围绕 0 |

**所有主信号必须按训练样本统计预先标准化**(z-score)再做延迟嵌入。标准化参数按股票 × 月度分组保存。

### 3.4 环境维定义

| 环境维名 | 定义 | 数据源 |
|---|---|---|
| `SPREAD` | 相对价差 `(ask - bid) / mid` | 快照 |
| `DEPTH5` | 五档总深度 `sum(V_bid_i + V_ask_i for i in 1..5)` | 快照 |
| `RV_PROXY` | 已实现波动代理(过去 30 秒 mid 变动平方和) | 快照 |
| `CANCEL_RATE` | 单位时间撤单数 / 单位时间挂单数 | 逐笔委托 |
| `LIFE_TIME` | 当前活跃挂单的中位存活时间 | 逐笔委托 |

环境维同样按股票 × 月度做标准化,与主信号同 scale。

### 3.5 τ 档位标定

τ 不能拍脑袋。三档定义:

| 档位 | 标定方法 |
|---|---|
| `short` | 每股票每月,主信号自相关首次过零的位置 / 2 |
| `mid` | 主信号自相关首次过零的位置 |
| `long` | 主信号互信息函数首次极小的位置 |

实际生产中,把每月标定的 τ 存到 `tau_calibration.parquet`,生产 pipeline 按日期查找。

### 3.6 H1 买卖分方向

对延迟嵌入类配置(范式 B/C 中的 OFI、CANCEL),**额外做买卖分方向**:

- 用 `OFI_NET_BUY` / `OFI_NET_SELL` 各跑一次 PD
- 用 `CANCEL_NET_BUY` / `CANCEL_NET_SELL` 各跑一次 PD

只产出方向化的 MAXPERS 因子,**不展开全套 13 统计量**(避免因子数爆炸)。

方向化 MAXPERS 因子额外 +X 个:OFI 簇 5 配置 × 2 方向 × 时段(每个配置取 FULL) = 10,CANCEL 簇 3 配置 × 2 方向 × FULL = 6。共 +16 个,**总因子数变 536**。

文档表头报"520",含方向后 "536"。两个数都对,看怎么算。

---

## 4. PD 统计量族(13 个,H0/H1 各算)

每个 PD 提取 13 个统计量。**H0 和 H1 各算一份,语义不同但定义相同**。

### 4.1 统计量定义

输入:持久性数组 `pers = death - birth`,以及对应的 `birth`、`death` 数组。
长度 `K = len(pers)`,排除 H0 的全局团(K-1)。

| 序号 | 统计量名 | 定义(伪代码) |
|---|---|---|
| 1 | `MAXPERS` | `max(pers)` |
| 2 | `TOP3_MEAN` | `mean(sorted(pers, reverse=True)[:3])` |
| 3 | `L2` | `sum(pers ** 2)` |
| 4 | `MAXPERS_RATIO` | `max(pers) / sum(pers)`,sum 为 0 时记 NaN |
| 5 | `ENT` | `-sum(p * log(p))` where `p = pers / sum(pers)` |
| 6 | `P75_MINUS_P25` | `np.percentile(pers, 75) - np.percentile(pers, 25)` |
| 7 | `SKEW` | `scipy.stats.skew(pers)` |
| 8 | `COUNT_LOW` | `(pers > 1.0 * theta_null).sum()` |
| 9 | `COUNT_HIGH` | `(pers > 3.0 * theta_null).sum()` |
| 10 | `MAXGAP` | `max(np.diff(sorted(pers, reverse=True)))`,K < 2 时记 NaN |
| 11 | `MEAN_BIRTH` | `mean(birth)` |
| 12 | `MEAN_DEATH` | `mean(death)` |
| 13 | `LIFETIME_OVER_BIRTH_MEAN` | `mean(pers / (birth + 1e-9))` |

### 4.2 边界情况处理

- `K == 0`(PD 为空):所有统计量返回 NaN
- `K == 1`:`SKEW`、`MAXGAP`、`P75_MINUS_P25` 返回 NaN,其他正常
- `sum(pers) == 0`:`MAXPERS_RATIO` 返回 NaN,`ENT` 返回 0

### 4.3 NaN 策略

因子值出现 NaN 时,**不做填充**,原样写入 parquet。下游回测自行处理。

---

## 5. 命名规则

### 5.1 主因子命名

```
GEO_PH_{CONFIG}_{HOMOLOGY}_{STAT}_{SEG}
```

- `CONFIG`:配置 ID,如 `C01`、`C02`
- `HOMOLOGY`:`H0` 或 `H1`
- `STAT`:13 个统计量名之一
- `SEG`:`FULL` 或 `OPEN`

示例:
- `GEO_PH_C01_H1_MAXPERS_FULL`
- `GEO_PH_C05_H0_ENT_OPEN`

### 5.2 买卖分方向因子命名

```
GEO_PH_{CONFIG}_H1_MAXPERS_{DIRECTION}_FULL
```

- `DIRECTION`:`BUY` 或 `SELL`

示例:
- `GEO_PH_C01_H1_MAXPERS_BUY_FULL`
- `GEO_PH_C15_H1_MAXPERS_SELL_FULL`

### 5.3 时段定义

| 时段 | 时间范围 | 说明 |
|---|---|---|
| `FULL` | 09:30:00 - 15:00:00 | 全天,排除 11:30-13:00 午休 |
| `OPEN` | 09:30:00 - 10:30:00 | 开盘小时,捕捉信息冲击下的循环 |

集合竞价(09:25-09:30)的 tick 数据**排除**,不参与点云构造。

---

## 6. 工程实现

### 6.1 目录结构

```
/home/hysheng/TDA/
├── ph_factors/
│   ├── __init__.py
│   ├── point_cloud.py         # 点云构造(范式 B/C 实现)
│   ├── signals.py             # 主信号 + 环境维计算
│   ├── landmark.py            # maxmin landmark 子采样(numba JIT)
│   ├── ph_kernels.py          # ripser 调用 + persistence 后处理
│   ├── statistics.py          # 13 个 PD 统计量
│   ├── factors.py             # 因子定义与注册
│   └── registry.py            # 沿用 Project4 注册模式
├── pipeline/
│   ├── ph_daily_prod.py       # 单日生产入口(generate_factors)
│   ├── ph_batch_run.py        # 多日批量
│   ├── consolidate_panel.py   # 面板汇总
│   └── tau_calibration.py     # τ 校准独立脚本
├── diagnostics/
│   ├── phase1_config_scan.py  # 阶段 1 配置粗扫
│   ├── theta_null_calib.py    # θ_null 标定
│   ├── pd_plot.py             # persistence diagram 可视化
│   └── attribution.py         # 高 H1 持久性的归因检查
├── tests/
│   ├── test_signals.py
│   ├── test_landmark.py
│   ├── test_ph_kernels.py
│   └── test_factors.py
└── configs/
    ├── pointcloud_configs.yaml  # 14 个配置定义
    ├── tau_calibration.parquet  # τ 月度标定结果
    └── theta_null.parquet       # θ_null 月度标定结果
```

### 6.2 核心数据结构

```python
@dataclass
class PointCloudCache:
    config_id: str                    # 配置 ID
    segment: str                      # FULL / OPEN
    direction: str | None             # None / BUY / SELL
    raw_cloud: np.ndarray             # (N, d) 原始点云
    landmarks: np.ndarray             # (M, d) landmark 点
    landmark_indices: np.ndarray      # (M,) landmark 在原点云的索引
    dist_matrix: np.ndarray           # (M, M) landmark 距离矩阵
    pd_h0: np.ndarray                 # (k0, 2) H0 (birth, death) 对
    pd_h1: np.ndarray                 # (k1, 2) H1 (birth, death) 对
    theta_null_h1: float              # 该股票该月 θ_null,从缓存读
    meta: dict                        # 嵌入参数、N、M 等

@dataclass
class FactorSpec:
    name: str                         # 完整因子名
    config_id: str
    homology: str                     # H0 / H1
    statistic: str                    # 13 个统计量之一
    segment: str                      # FULL / OPEN
    direction: str | None             # None / BUY / SELL
```

### 6.3 单股单日 pipeline 流程

```
def process_stock_day(stock_id, date):
    # 1. 加载原始数据
    snapshot = load_snapshot(stock_id, date)
    order_flow = load_order_flow(stock_id, date)
    trade = load_trade(stock_id, date)

    # 2. 计算所有主信号 + 环境维
    signals = compute_signals(snapshot, order_flow, trade)
    # signals = dict(OFI_NET, OB_IMBAL, DP_MICRO, CANCEL_NET, ...)

    # 3. 加载 τ 标定 + θ_null
    tau_lookup = load_tau_calibration(stock_id, date.month)
    theta_null = load_theta_null(stock_id, date.month)

    # 4. 对每个配置 × 时段 × 方向 算 PD
    cache_list = []
    for config in CONFIGS:  # 14 个
        for seg in config.h1_segments:  # FULL / OPEN
            for direction in get_directions(config):  # None / BUY / SELL
                cloud = build_point_cloud(config, seg, direction, signals, tau_lookup)
                landmarks, idx, dist = maxmin_landmark(cloud, M=500)
                pd_h0, pd_h1 = ripser_compute(dist, maxdim=1)
                cache_list.append(PointCloudCache(...))

    # 5. 提取所有因子
    factor_dict = {}
    for spec in ALL_FACTOR_SPECS:  # 520 + 16 = 536
        cache = lookup_cache(cache_list, spec)
        factor_dict[spec.name] = compute_statistic(cache, spec)

    # 6. 写 parquet
    save_factors(factor_dict, stock_id, date)
```

### 6.4 多进程架构

完全沿用 Project3/4 模式:

- 启动方式:`forkserver`
- `_worker_init`:加载只读资源(τ 标定、θ_null、配置 YAML)
- 池批量化:每 200 股重启 worker(防 ripser 内存碎片)
- 单股 timeout:60s,超时写 `failure_log` 不阻塞
- 断点续跑:`per-stock` 输出文件检测,沿用 Project5 模式

### 6.5 性能预算

| 阶段 | 单股单日耗时(估算) |
|---|---|
| 信号 + 环境维计算 | 0.2s(逐笔聚合,有缓存) |
| 14 配置 × 平均 1.5 段 × 平均 1.3 方向 ≈ 27 次 PD | 27 × 0.8s = 21.6s |
| 因子后处理(536 个,O(K) 轻量) | 0.1s |
| **合计** | **~22s/股/日** |

**对比 Project4 的 3.6s/日,TDA 必须异步独立 pipeline 生产**,T+1 延迟可接受。

### 6.6 关键依赖

```
ripser >= 0.6
persim >= 0.3
scikit-learn (kNN, 备用)
numba (landmark JIT)
numpy, scipy, pandas
pyarrow (parquet)
```

### 6.7 输出格式

完全对齐 Project3/4:

```
/factors/PH/{date}/{factor_name}.parquet
```

每个 parquet 文件包含 columns: `[stock_id, factor_value]`,index 为股票代码。

**单入口 API**:

```python
def generate_factors(date: str) -> pd.DataFrame:
    """
    生产指定日期的所有 536 个因子。

    Args:
        date: 'YYYYMMDD' 格式

    Returns:
        DataFrame, index=stock_id, columns=536 个因子名
    """
```

---

## 7. 三阶段生产路线

### 阶段 1:配置粗扫(用于淘汰废配置)

**目标**:在 14 个配置里筛掉 4-6 个废配置,避免在阶段 2 浪费算力。

**配置**:
- 股票样本:100 只(主板/创业板/科创板 + 大中小市值各档)
- 时间区间:最近 60 个交易日
- 只算 1 个因子:`GEO_PH_{CONFIG}_H1_MAXPERS_FULL`
- 不计算 H0、不计算 OPEN 段、不计算方向

**诊断指标**:
- 截面均值是否显著 > θ_null
- 跨日时间序列的稳定性(自相关、变异系数)
- 与基准信号的相关性(确保不是噪声)

**决策规则**:
- MAXPERS 均值 < 1.5 × θ_null:废配置,砍掉
- MAXPERS 跨日 IQR / median > 2:不稳定,砍掉
- 同簇内多个配置相关性 > 0.95:留 alpha 信号更强的一个

预期保留 8-10 个配置进入阶段 2。

**交付物**:`diagnostics/phase1_config_scan.py` 的输出报告 + 胜出配置清单。

### 阶段 2:因子展开(全因子集小样本验证)

**目标**:在胜出配置上展开全 536 因子,做相关性筛选 + IC 初评。

**配置**:
- 股票样本:阶段 1 同一批 100 只
- 时间区间:60 个交易日
- 展开:胜出配置 × 13 统计量 × H0/H1 × 时段 × 方向 ≈ 320-400 个因子

**审计流程**:
1. 因子值合理性:NaN 率 < 30%、方差 > 0、不全恒定
2. **截面相关性矩阵**:与 Project4 12 个蓝因子 + GaFeatureSample 库比较。相关性 > 0.85 的剔除。
3. **IC / ICIR 初评**:用 BackTester + CNE5 中性化,Rank IC 显著性 > 1.96
4. **机制归因**:对 H1 高持久性时段,实际检视该股的挂撤序列,验证可识别循环存在

**交付物**:白名单因子清单(预期 15-25 个),回测报告。

### 阶段 3:全样本生产

**目标**:对白名单因子做全样本 × 6 年生产,产出最终因子面板。

**配置**:
- 股票样本:全 A 股 ~5000 只
- 时间区间:6 年历史
- 因子集:阶段 2 白名单

**性能**:异步独立 pipeline 隔夜跑,与 Project4 主线程解耦。

**交付物**:
- `/factors/PH/` 完整面板
- 整理后的 `generate_factors(date)` 单入口
- Production-ready zip 包,对齐 Project4 交付标准

---

## 8. 关键质量保证

### 8.1 单元测试要求

每个核心模块必须有测试:

- `test_signals.py`:主信号计算正确性,边界情况(空数据、单 tick 等)
- `test_landmark.py`:maxmin 算法选点正确性、距离矩阵对称性
- `test_ph_kernels.py`:ripser 调用正确性,PD 数据结构合规
- `test_factors.py`:13 统计量在已知 PD 上的数值正确性

### 8.2 集成测试

构造一个**已知存在 H1 环的合成点云**(如 noisy circle),验证全 pipeline 能正确识别 MAXPERS、ENT 等因子值在合理范围。

### 8.3 数值稳定性

- 所有除法操作:分母 + `1e-9` 防 0
- 所有 `log`:`log(max(x, 1e-12))` 防 0
- NaN 不填充,但代码必须能 handle NaN 输入(不抛异常)

### 8.4 复现性

- ripser 默认是确定性的,但 maxmin landmark **必须固定随机种子**:`seed = hash(stock_id + date)`
- numba JIT 函数中如有随机操作,显式设种子
- 不要使用模块级全局 RNG(踩过 Project4 的坑)

---

## 9. 给 Claude Code 的实现建议

### 9.1 优先实现顺序

1. **`signals.py`**(基础,所有上游依赖此)
2. **`landmark.py`** + 测试(性能命脉,先打牢)
3. **`ph_kernels.py`** + 测试(包装 ripser,做 PD 后处理)
4. **`statistics.py`** + 测试(13 个统计量,纯函数,易测)
5. **`point_cloud.py`**(组装点云,调用上述)
6. **`factors.py`** + `registry.py`(注册全 536 个 FactorSpec)
7. **`pipeline/ph_daily_prod.py`**(单股单日入口)
8. **`diagnostics/phase1_config_scan.py`**(立刻可用于阶段 1)
9. **`pipeline/ph_batch_run.py`** + `consolidate_panel.py`

### 9.2 易踩的坑

- **ripser 输入必须是 `np.float64`**,float32 会触发奇怪报错
- **distance_matrix=True 时**,输入矩阵必须严格对称,且对角线为 0
- **ripser 输出的 PD 包含一个 `(0, inf)` 项**(全局连通团),H0 处理时必须显式过滤
- **numba JIT 函数**:第一次调用会 warmup,在多进程下每个 worker 都要 warmup 一次,首批股可能慢
- **maxmin landmark**:第一个点的选择(随机 vs 质心 vs 第一个点)会显著影响后续结果,务必固定策略并加种子
- **τ 单位**:统一用 tick 数,不要用秒数

### 9.3 必读对比文档

实现时请对照以下 Project4 已有代码(假定已在仓库可访问):

- `Project4/geo_factors/registry.py`:因子注册模式,TDA 完全沿用
- `Project4/geo_factors/fast_kernels.py`:numba JIT 写法参考
- `Project4/pipeline/geo_daily_prod.py`:单日生产入口架构
- `Project4/pipeline/geo_batch_run.py`:多进程 + forkserver + 池批量化

如发现 Project4 模式和本文档冲突,优先按本文档,然后告诉我冲突点。

---

## 10. 不在本期范围(明确排除)

为防止 Claude Code 过度发挥,以下功能本期**不实现**:

- 范式 A(多变量同时刻嵌入)—— 论证不站,已砍
- H2 及以上同调 —— 计算爆炸,金融含义弱
- Persistence landscape / image 向量化因子 —— 留待阶段 3 后扩展
- 跨股票拓扑相似度 —— 后续 Project7
- Mapper 算法 —— 后续扩展
- 跨日动态因子(滚动均值、Z-score 等) —— 在阶段 2 alpha 验证后统一加
- 神经网络/ML 接入 —— 单纯产出因子值,下游建模解耦

---

## 11. 验收标准

Claude Code 实现完成后,需通过以下验收:

- [ ] 所有单元测试通过
- [ ] 集成测试(合成 noisy circle)正确识别 H1 环
- [ ] 跑 10 只股票 5 个交易日,产出 536 个因子,NaN 率 < 30%
- [ ] 单股单日耗时 < 30s(允许略超 22s 估算)
- [ ] 复现性:同 seed 两次跑结果完全一致
- [ ] `generate_factors(date)` 单入口可调用
- [ ] 输出 parquet 格式对齐 Project4

---

## 附录 A:配置 YAML 模板

`configs/pointcloud_configs.yaml` 结构示例:

```yaml
configs:
  - id: C01
    cluster: OFI
    paradigm: B
    main_signal: OFI_NET
    tau_level: short
    env_dims: []
    total_dim: 2
    embed_order: 2
    h1_segments: [FULL, OPEN]
    h0_segments: [FULL, OPEN]
    directions: [null, BUY, SELL]  # null=合并; BUY/SELL 只产 MAXPERS

  - id: C04
    cluster: OFI
    paradigm: C
    main_signal: OFI_NET
    tau_level: mid
    env_dims: [SPREAD]
    total_dim: 3
    embed_order: 2
    h1_segments: [FULL]
    h0_segments: [FULL]
    directions: [null]

  # ... 其他 12 个配置
```

## 附录 B:完整因子清单生成代码

```python
def enumerate_all_factors(configs):
    """从 14 个配置展开全 536 个 FactorSpec。"""
    STATS = ["MAXPERS", "TOP3_MEAN", "L2", "MAXPERS_RATIO",
             "ENT", "P75_MINUS_P25", "SKEW",
             "COUNT_LOW", "COUNT_HIGH", "MAXGAP",
             "MEAN_BIRTH", "MEAN_DEATH", "LIFETIME_OVER_BIRTH_MEAN"]
    specs = []
    for cfg in configs:
        # 主因子(13 统计量 × H0/H1 × 时段)
        for homology, segs in [("H1", cfg.h1_segments), ("H0", cfg.h0_segments)]:
            for stat in STATS:
                for seg in segs:
                    specs.append(FactorSpec(
                        name=f"GEO_PH_{cfg.id}_{homology}_{stat}_{seg}",
                        config_id=cfg.id, homology=homology,
                        statistic=stat, segment=seg, direction=None))
        # 方向化 MAXPERS(仅 H1 FULL)
        if cfg.directions and len(cfg.directions) > 1:
            for direction in ["BUY", "SELL"]:
                specs.append(FactorSpec(
                    name=f"GEO_PH_{cfg.id}_H1_MAXPERS_{direction}_FULL",
                    config_id=cfg.id, homology="H1",
                    statistic="MAXPERS", segment="FULL", direction=direction))
    return specs

# 预期 len(enumerate_all_factors(CONFIGS)) == 536
```

---

**本文档结束。任何执行歧义,反馈给 Michael 后再决策,不要自行扩展或裁剪。**