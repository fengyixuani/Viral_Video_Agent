"""Lightweight rotating logger for pipeline observability.

统一写到 ``logs/app.log``（滚动，最多 5 个 2MB 文件），同时输出到 stderr。
每条日志带毫秒时间戳、级别、模块。关键流程（HTTP 请求、LLM/视觉调用、工具
编排、缓存命中、错误）都会打点，方便回溯和 debug。

用法::

    from obs import get_logger, new_request_id
    log = get_logger("orchestrator")
    rid = new_request_id()
    log.info("[%s] analyze start uri=%s", rid, uri)
"""
from __future__ import annotations

import logging
import os
import uuid
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_configured = False


def _configure():
    global _configured
    if _configured:
        return
    root = logging.getLogger("vva")
    root.setLevel(getattr(logging, _LEVEL, logging.INFO))
    root.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"vva.{name}")


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]
