"""
_bootstrap.py
=============
统一 sys.path 引导，沿用 Project4 模式。
每个入口脚本开头两行固定写法：
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import _bootstrap  # noqa
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

for _sub in ("ph_factors", "pipeline", "diagnostics"):
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Processor 包位于 /home/hysheng/Processor，需要把父目录加入 sys.path
_HYSHENG = "/home/hysheng"
if os.path.isdir(_HYSHENG) and _HYSHENG not in sys.path:
    sys.path.append(_HYSHENG)
