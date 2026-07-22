"""pytest 引导：把 src/ 与 src/shared/ 加入 sys.path。

三人协作拆分后，owned 包（understanding/editing/drama）在 src/ 下，
共享层（as_core/obs/tools/inputs/...）在 src/shared/ 下，两者都要在 path 上，
扁平 import（import as_core / from tools import ...）才能解析。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src", "shared"), os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
