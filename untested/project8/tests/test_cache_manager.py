import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/hysheng/project8')

import data.cache_manager as cm


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'CACHE_ROOT', str(tmp_path))


def test_save_and_load():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
    cm.save(df, 'test_ns', 'test_key')
    assert cm.exists('test_ns', 'test_key')
    loaded = cm.load('test_ns', 'test_key')
    pd.testing.assert_frame_equal(df, loaded)


def test_exists_false_before_save():
    assert not cm.exists('test_ns', 'nonexistent')


def test_load_or_compute_caches():
    call_count = [0]

    def fn():
        call_count[0] += 1
        return pd.DataFrame({'x': [1, 2]})

    result1 = cm.load_or_compute('ns2', 'k1', fn)
    result2 = cm.load_or_compute('ns2', 'k1', fn)
    assert call_count[0] == 1  # fn 只被调用一次
    pd.testing.assert_frame_equal(result1, result2)
