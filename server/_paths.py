"""共享路径与 sys.path 设置（app.py 与各 routes_*.py 都从这里取）。

导入本模块即完成：
1. `config.env` 加载：读取项目根 `config.env`（如果存在），把 KEY=VALUE 用
   `os.environ.setdefault` 注入——已 export 的环境变量优先级更高；支持 ``${VAR}`` 引用
   同文件前面已定义的键或已 export 的环境变量。所有下游 `os.getenv(...)` 自动生效。
2. `sys.path` 设置：owned 包（understanding/orchestration/editing/drama）在 `src/` 下，
   共享层在 `src/shared/` 下，两者都加入 sys.path，保持扁平 import。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
SHARED_DIR = os.path.join(SRC_DIR, "shared")


def _load_config_env() -> None:
    """把项目根 config.env 里的 KEY=VALUE 注入 os.environ（不覆盖已存在的 env）。"""
    path = os.path.join(ROOT, "config.env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for raw in stream:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key:
                    continue
                # ${VAR} 展开：可引用同文件前面已注入的键或已 export 的环境变量
                value = os.path.expandvars(value)
                os.environ.setdefault(key, value)
    except OSError:
        pass


_load_config_env()

for _p in (SHARED_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
