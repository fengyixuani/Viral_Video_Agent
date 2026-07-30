"""pipeline_utils (Agent 移植版薄封装).

原 whq 依赖 Split 仓 ``common/pipeline_utils`` 的 ``ask_qianfan`` / ``loads_with_repair``。
这里只补齐 whq_clone 实际用到的这两个函数：
- ``ask_qianfan``: 走 Agent 的 wenchain 网关（复用 ``as_core`` 的 URL/KEY/MAX_TOKENS 配置，
  单一配置源），同步 ``requests`` POST，签名/返回与原版保持一致 ``(content, data)``。
- ``loads_with_repair``: 抽取并解析 JSON，可选 json_repair 兜底。

whq 的 edit_planner / shot_matcher / voiceover 通过 ``from pipeline_utils import ...`` 引用；
因 ``_common`` 已把本目录置于 sys.path 首位，会解析到此文件而非 Split 的同名模块。
"""
import hashlib
import json
import os
import re
import time

import requests

# 复用 as_core 的网关配置作为单一配置源（同一个 wenchain 网关、同一个 key）。
import as_core

try:  # 可选：JSON 修复兜底
    import json_repair as _json_repair
except ImportError:  # pragma: no cover
    _json_repair = None


def ask_qianfan(messages, model=None, max_tokens=None, temperature=0.2, timeout=None):
    """同步调用 wenchain 网关的 chat/completions，返回 ``(content_text, full_json)``。

    与 Split 版签名一致。默认强制 ``response_format=json_object``（whq 所有调用都要 JSON）。
    """
    timeout = timeout or int(os.getenv("LLM_TIMEOUT", os.getenv("QIANFAN_TIMEOUT", "300")))
    base_url = (as_core.WENCHAIN_BASE_URL or "").rstrip("/")
    api_key = as_core.WENCHAIN_API_KEY
    if not api_key:
        raise RuntimeError("WENCHAIN_API_KEY 未配置（whq_clone LLM 调用需要网关）")
    payload = {
        "model": model or as_core.TEXT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.8,
        "max_tokens": max_tokens or as_core.MAX_TOKENS,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    attempts = max(1, int(os.getenv("QIANFAN_RETRY_COUNT", "3")))
    retry_sleep = float(os.getenv("QIANFAN_RETRY_SLEEP", "2"))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                base_url + "/chat/completions",
                headers={"Authorization": "Bearer {}".format(api_key),
                         "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"], data
        except (requests.HTTPError, requests.RequestException) as exc:
            body = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            status = exc.response.status_code if getattr(exc, "response", None) is not None else "?"
            last_error = RuntimeError("HTTP {}: {}".format(status, body))
            retryable = (
                isinstance(exc, requests.RequestException) and not isinstance(exc, requests.HTTPError)
            ) or any(tok in body.lower() for tok in
                     ("connection reset", "conn talk failed", "llm_rr_error"))
            if attempt < attempts and retryable:
                print("QIANFAN_RETRY {}/{}: {}".format(attempt, attempts, str(last_error)[-300:]), flush=True)
                time.sleep(retry_sleep)
                continue
            raise last_error from exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("ask_qianfan failed without a captured error")


def strip_json_noise(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.replace("\ufeff", "").strip()


def extract_json_text(text):
    if "## 模型输出" in text:
        text = text.split("## 模型输出", 1)[1]
    text = strip_json_noise(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object")
    return text[start:end + 1]


def loads_with_repair(text):
    """解析 JSON，失败时用 json_repair 兜底；两者都失败才抛原始 JSONDecodeError。"""
    raw = extract_json_text(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        if _json_repair is None:
            raise
        try:
            return _json_repair.loads(raw)
        except Exception:
            raise exc


def load_jsonish(path):
    with open(path, "r", encoding="utf-8") as f:
        return loads_with_repair(f.read())


# ---------- 阶段缓存(可续跑) ----------

def fingerprint(*parts):
    """把任意可 JSON 化的输入摘成短指纹, 用于判断阶段缓存是否还有效。"""
    h = hashlib.sha1()
    for p in parts:
        try:
            h.update(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str).encode())
        except (TypeError, ValueError):
            h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def env_fingerprint(names):
    """按名单取环境变量组成指纹片段: 影响该阶段结果的 env 变了就让缓存失效。"""
    return {n: os.getenv(n, "") for n in names}


def file_fingerprint(paths):
    """文件列表的 (路径, 大小, mtime) 指纹: 输入素材换了就让缓存失效。"""
    out = []
    for p in paths or []:
        try:
            st = os.stat(p)
            out.append([os.path.abspath(p), st.st_size, int(st.st_mtime)])
        except OSError:
            out.append([str(p), -1, -1])
    out.sort()
    return out


def load_stage(path, fp):
    """读阶段缓存; 指纹不符/文件缺失/WHQ_RESUME=0 时返回 None(需要重跑)。"""
    if os.getenv("WHQ_RESUME", "1") in ("0", "false", "False"):
        return None
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    if obj.get("_fingerprint") != fp:
        return None
    return obj


def save_stage(path, fp, payload):
    """落阶段缓存(payload 里挂 _fingerprint)。写失败不抛, 不阻断主流程。"""
    obj = dict(payload)
    obj["_fingerprint"] = fp
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        print("[pipeline_utils] 阶段缓存写入失败(忽略): {}".format(str(exc)[:120]), flush=True)
    return path


def parallel_map(fn, items, workers=None, env_var="WHQ_LLM_CONCURRENCY", default=3):
    """按序返回 [fn(x) for x in items], 但并发执行(LLM/VLM 调用是网络等待型)。

    workers<=1 时退化为串行。fn 抛异常时该项返回该异常对象, 由调用方决定如何降级
    (whq 各处的约定都是「单点失败不阻断主流程」)。
    """
    items = list(items)
    if not items:
        return []
    if workers is None:
        workers = int(os.getenv(env_var, str(default)))
    workers = max(1, min(workers, len(items)))
    if workers == 1:
        results = []
        for x in items:
            try:
                results.append(fn(x))
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
        return results
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, x) for x in items]
        out = []
        for fu in futures:
            try:
                out.append(fu.result())
            except Exception as exc:  # noqa: BLE001
                out.append(exc)
    return out

