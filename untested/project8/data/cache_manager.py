"""data/cache_manager.py — 中间结果 parquet 缓存，支持断点续算。"""
import os
import pandas as pd


CACHE_ROOT = "/home/hysheng/project8/output/cache"


def cache_path(namespace: str, key: str) -> str:
    return os.path.join(CACHE_ROOT, namespace, f"{key}.parquet")


def exists(namespace: str, key: str) -> bool:
    return os.path.exists(cache_path(namespace, key))


def save(df: pd.DataFrame, namespace: str, key: str) -> None:
    path = cache_path(namespace, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)


def load(namespace: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(cache_path(namespace, key))


def load_or_compute(namespace: str, key: str, fn) -> pd.DataFrame:
    """若缓存存在直接读取，否则调用 fn() 计算并缓存。"""
    if exists(namespace, key):
        return load(namespace, key)
    df = fn()
    save(df, namespace, key)
    return df
