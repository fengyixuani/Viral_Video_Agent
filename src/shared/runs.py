"""运行产物登记：把每次运行（复刻编排 / Agent 剪辑 / AI 短剧）产物落到**单独一个可读命名的
run 目录**，并维护一个 `outputs/index.json` 索引，方便"这次是哪个 / 找历史"。

run 目录命名：``outputs/{YYYYMMDD_HHMMSS}_{kind}[_{项目slug}]_{rid6}/``
- 时间戳前缀 → `ls outputs` 天然按时间排序、一眼看出最新一次；
- kind ∈ replicate（编排）/ agentcut（Agent 剪辑）/ drama（短剧）。

索引 `outputs/index.json`：``[{run_id, kind, project, created_at, created_at_str, dir, ...extra}]``，
最新在前。`record()` 幂等更新（按 run_id）。
"""
import json
import os
import re
import time
import uuid
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")
INDEX_PATH = os.path.join(OUTPUTS_ROOT, "index.json")


def _slug(text: str) -> str:
    s = re.sub(r"[\s/\\]+", "_", (text or "").strip())
    s = re.sub(r"[^\w\u4e00-\u9fff-]", "", s)
    return s[:40]


def _rel(path: str) -> str:
    """相对 outputs/ 之上的项目根的相对路径，便于跨机可读、也可拼 /outputs URL。"""
    try:
        r = os.path.relpath(os.path.abspath(path), PROJECT_ROOT)
        return r if not r.startswith("..") else path
    except ValueError:
        return path


def new_run(kind: str, project: str = "") -> dict:
    """创建并返回一个 run 目录信息 dict：{run_id, kind, project, created_at, created_at_str, dir}。"""
    ts = time.time()
    stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
    rid = uuid.uuid4().hex[:6]
    slug = _slug(project)
    name = f"{stamp}_{kind}" + (f"_{slug}" if slug else "") + f"_{rid}"
    run_dir = os.path.join(OUTPUTS_ROOT, name)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_id": name,
        "kind": kind,
        "project": project,
        "created_at": ts,
        "created_at_str": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
        "dir": run_dir,
    }


def _load_index() -> list:
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def record(run: dict, **extra) -> None:
    """把 run 记录写进 outputs/index.json（按 run_id 幂等更新，最新在前）。失败不抛。"""
    if not run or not run.get("run_id"):
        return
    entry = {
        "run_id": run["run_id"],
        "kind": run.get("kind", ""),
        "project": run.get("project", ""),
        "created_at": run.get("created_at") or time.time(),
        "created_at_str": run.get("created_at_str") or "",
        "dir": _rel(run.get("dir", "")),
    }
    for k, v in extra.items():
        entry[k] = _rel(v) if (isinstance(v, str) and ("/" in v or v.endswith(".json") or v.endswith(".mp4"))) else v
    try:
        os.makedirs(OUTPUTS_ROOT, exist_ok=True)
        index = [e for e in _load_index() if e.get("run_id") != entry["run_id"]]
        index.append(entry)
        index.sort(key=lambda e: e.get("created_at") or 0, reverse=True)
        tmp = INDEX_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, INDEX_PATH)
    except OSError:
        pass


def list_runs(limit: int = None, kind: str = "") -> list:
    runs = _load_index()
    if kind:
        runs = [r for r in runs if r.get("kind") == kind]
    return runs[:limit] if limit else runs
