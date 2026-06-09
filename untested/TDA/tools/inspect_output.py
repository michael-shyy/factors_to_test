"""查看单日因子输出结果。

两种用法：
    # 1. 作为模块被 time_one_stock.py 调用（无需 pkl，直接打印）
    python tools/time_one_stock.py sh600000 20250507

    # 2. 读 pkl 文件（批量生产后）
    python tools/inspect_output.py output/factors_daily_pkl/factors_20250507.pkl
"""
import os
import sys
import numpy as np
import pandas as pd


def print_report(df: pd.DataFrame, source: str = ""):
    factor_cols = [c for c in df.columns if c not in ("date", "cn_code")]

    print(f"\n{'='*60}")
    if source:
        print(f"来源: {source}")
    print(f"shape: {df.shape[0]} 只股票 × {len(factor_cols)} 个因子")
    print(f"{'='*60}")

    print(f"\n因子数: {len(factor_cols)}")
    if "date" in df.columns:
        print(f"日期:   {df['date'].iloc[0]}")
    if "cn_code" in df.columns:
        print(f"股票:   {df['cn_code'].tolist()[:10]}{'...' if len(df) > 10 else ''}")

    # NaN 统计
    nan_counts = df[factor_cols].isna().sum()
    nan_rate   = nan_counts / len(df)
    print(f"\nNaN 统计:")
    print(f"  全为 NaN 的因子数:      {(nan_counts == len(df)).sum()}")
    print(f"  NaN 率 > 50% 的因子数:  {(nan_rate > 0.5).sum()}")
    print(f"  NaN 率 > 30% 的因子数:  {(nan_rate > 0.3).sum()}")
    print(f"  平均 NaN 率:            {nan_rate.mean():.1%}")

    valid = df[factor_cols].dropna(axis=1, how="all")
    print(f"\n有效因子数（至少一个非NaN）: {len(valid.columns)}")

    # 抽样 MAXPERS
    sample = [c for c in factor_cols
              if "MAXPERS" in c and "H1" in c and "FULL" in c
              and "BUY" not in c and "SELL" not in c][:8]
    if sample:
        print(f"\n抽样因子值（H1_MAXPERS_FULL）:")
        show_cols = (["cn_code"] if "cn_code" in df.columns else []) + sample
        print(df[show_cols].to_string(index=False))

        # τ/z-score 诊断
        vals = [df[c].iloc[0] for c in sample if not df[c].isna().all()]
        if len(vals) >= 2:
            ratio = max(vals) / (min(vals) + 1e-12)
            if ratio < 1.5:
                print(f"\n  ⚠  各配置 MAXPERS 差异很小 (max/min={ratio:.1f}x)")
                print(f"     τ 标定或 z-score 可能未生效，建议带 --calib_dir 重跑")
            else:
                print(f"\n  ✓  各配置 MAXPERS 有差异 (max/min={ratio:.1f}x)，τ 生效")

    # 按配置 NaN 率
    print(f"\n按配置 NaN 率:")
    for cfg in ["C01","C02","C03","C04","C05",
                "C07","C08","C09",
                "C11","C12","C13","C14",
                "C15","C16","C17"]:
        cols = [c for c in factor_cols if f"_{cfg}_" in c]
        if cols:
            r = df[cols].isna().mean().mean()
            bar = "█" * int(r * 20)
            print(f"  {cfg}: {r:5.1%}  {bar:20s}  ({len(cols)} 个因子)")

    print("=" * 60 + "\n")


# ── 主逻辑（直接执行时读 pkl；作为模块时只用 print_report）───────────────────

if __name__ == "__main__":
    import pickle

    _PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DEFAULT     = os.path.join(_PROJECT_DIR, "output", "factors_daily_pkl",
                                "factors_20240102.pkl")
    arg = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT

    if not os.path.exists(arg):
        print(f"找不到文件: {arg}")
        print(f"用法: python tools/inspect_output.py <pkl路径>")
        sys.exit(1)

    with open(arg, "rb") as f:
        df = pickle.load(f)

    print_report(df, source=arg)
