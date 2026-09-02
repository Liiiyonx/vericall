# -*- coding: utf-8 -*-
"""测试环境引导：把 src 与 evaluation 加入 import 路径。

这样测试里可以直接 `from fusion.fusion_orchestrator import ...`
以及 `from metrics import ...`，无需手动改 PYTHONPATH。
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in ("src", "evaluation"):
    _full = str(_ROOT / _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)
