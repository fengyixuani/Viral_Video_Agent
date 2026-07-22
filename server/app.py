"""标准库 HTTP 服务入口（路由已按模块拆分，见 server/routes_*.py）。

- 路径/`sys.path` 设置在 `_paths.py`。
- 各业务路由分散在 `routes_common / routes_understanding / routes_editing / routes_drama`，
  每个模块导出 ``GET`` / ``POST`` 两个 {路径: 处理函数(handler)} 字典；本文件只做聚合与分发。
- Handler 只保留跨模块的基础设施：JSON/文件响应、multipart 上传解析、SSE 发送、路径解析。
"""
import asyncio
import json
import mimetypes
import os
import socket
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import _paths  # noqa: F401  导入即完成 sys.path 设置（shared + src）
from _paths import ROOT, STATIC_DIR, UPLOAD_DIR
import obs
from inputs import load

import routes_common
import routes_understanding
import routes_orchestration
import routes_editing
import routes_drama
import routes_debug

_log = obs.get_logger("http")

# 聚合各模块路由表：{path: fn(handler)}
GET_ROUTES = {}
POST_ROUTES = {}
for _m in (routes_common, routes_understanding, routes_orchestration, routes_editing, routes_drama, routes_debug):
    GET_ROUTES.update(getattr(_m, "GET", {}) or {})
    POST_ROUTES.update(getattr(_m, "POST", {}) or {})


class Handler(BaseHTTPRequestHandler):
    server_version = "ViralVideoAgent/1.0"

    # ---- 基础设施：请求/响应 ----
    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(length) if length else b""

    def _payload(self):
        try:
            return json.loads(self._read_body() or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path):
        if not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as stream:
            body = stream.read()
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        # index.html / 静态资源不允许强缓存——迭代频繁，防止浏览器加载到旧版 JS 导致 UI 与后端不同步
        if path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _find_by_digest(self, digest: str):
        """按 <digest>_ 前缀在 uploads 目录里查已有文件，返回文件名或 None。"""
        if not digest:
            return None
        prefix = digest + "_"
        for existing in os.listdir(UPLOAD_DIR):
            if existing.startswith(prefix) or existing.startswith(digest + "."):
                return existing
        return None

    def _resolve_local(self, uri: str) -> str:
        """把 uploads/xxx 或相对/绝对路径解析成本地存在的文件路径。"""
        if not uri:
            return ""
        if os.path.isabs(uri) and os.path.isfile(uri):
            return uri
        for cand in (uri, os.path.join(ROOT, uri)):
            if os.path.isfile(cand):
                return cand
        return ""

    # ---- 基础设施：SSE ----
    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _emit(self, event):
        self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _sse_done(self):
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse(self, make_gen):
        """通用 SSE：从请求体 load 出 InputBundle，逐事件回传（analyze / replicate 用）。"""
        payload = self._payload()
        self._sse_headers()
        try:
            bundle = load(payload)

            async def drain():
                async for event in make_gen(bundle):
                    self._emit(event)

            asyncio.run(drain())
        except Exception as exc:  # noqa: BLE001
            _log.error("SSE stream failed: %s", exc, exc_info=True)
            traceback.print_exc()
            try:
                self._emit({"type": "error", "message": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return
        self._sse_done()

    # ---- 分发 ----
    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        fn = GET_ROUTES.get(path)
        if fn:
            return fn(self)
        if path == "/":
            return self._serve_file(os.path.join(STATIC_DIR, "index.html"))
        if path.startswith("/static/"):
            candidate = os.path.realpath(os.path.join(STATIC_DIR, path[len("/static/"):]))
            if candidate.startswith(os.path.realpath(STATIC_DIR) + os.sep):
                return self._serve_file(candidate)
            return self.send_error(403)
        if path.startswith("/uploads/"):
            candidate = os.path.realpath(os.path.join(ROOT, path.lstrip("/")))
            if candidate.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
                return self._serve_file(candidate)
            return self.send_error(403)
        if path.startswith("/outputs/"):
            outputs_dir = os.path.realpath(os.path.join(ROOT, "outputs"))
            candidate = os.path.realpath(os.path.join(ROOT, path.lstrip("/")))
            if candidate.startswith(outputs_dir + os.sep):
                return self._serve_file(candidate)
            return self.send_error(403)
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        fn = POST_ROUTES.get(path)
        if fn:
            return fn(self)
        self.send_error(404)

    def log_message(self, fmt, *args):
        _log.info("%s %s", self.address_string(), fmt % args)


def _port_in_use(host: str, port: int) -> bool:
    """探测 (host, port) 是否已被占用。"""
    probe_host = host or "0.0.0.0"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # 不设 SO_REUSEADDR：真实占用（有进程 listen）时 bind 会失败，正是我们要检测的
        try:
            sock.bind((probe_host, port))
            return False
        except OSError:
            return True


def _find_free_port(host: str, start: int, tries: int = 50) -> int:
    """从 start 起找一个可用端口；都不行则让系统随机分配一个。"""
    for candidate in range(start, start + tries):
        if candidate <= 65535 and not _port_in_use(host, candidate):
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host or "0.0.0.0", 0))
        return sock.getsockname()[1]


def main():
    want_port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    # 端口检查：选定端口被占用则自动换一个可用端口，并在命令行提示
    port = want_port
    if _port_in_use(host, want_port):
        port = _find_free_port(host, want_port + 1)
        print(f"[端口检查] {want_port} 端口被占用，自动改为 {port} 端口", flush=True)
        _log.warning("port %d in use, switched to %d", want_port, port)
    server = ThreadingHTTPServer((host, port), Handler)
    shown = host if host != "0.0.0.0" else "0.0.0.0（本机所有网卡）"
    print(f"Viral Video Agent running at http://{shown}:{port}", flush=True)
    _log.info("server listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
