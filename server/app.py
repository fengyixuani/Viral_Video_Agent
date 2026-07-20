"""Standard-library HTTP service for the viral-video replication workbench."""
import asyncio
import hashlib
import json
import mimetypes
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

import skills as skill_mod
import trends
import obs
import connector
import projects
from agent_edit import loop as agent_edit_loop
from agent_edit import tools as agent_edit_tools
from agent import ReplicationAgent
from inputs import load

AGENT = ReplicationAgent()
_log = obs.get_logger("http")


class Handler(BaseHTTPRequestHandler):
    server_version = "ViralVideoAgent/1.0"

    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(length) if length else b""

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

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._serve_file(os.path.join(STATIC_DIR, "index.html"))
        elif path == "/api/skills":
            self._json({"skills": skill_mod.skill_list()})
        elif path == "/api/projects":
            self._json({"projects": projects.list_projects()})
        elif path == "/api/agent_edit/tools":
            self._json({"tools": agent_edit_tools.tool_schemas()})
        elif path.startswith("/static/"):
            candidate = os.path.realpath(os.path.join(STATIC_DIR, path[len("/static/"):]))
            if candidate.startswith(os.path.realpath(STATIC_DIR) + os.sep):
                self._serve_file(candidate)
            else:
                self.send_error(403)
        elif path.startswith("/uploads/"):
            candidate = os.path.realpath(os.path.join(ROOT, path.lstrip("/")))
            if candidate.startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
                self._serve_file(candidate)
            else:
                self.send_error(403)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/upload":
            self._upload()
        elif path == "/api/upload/check":
            self._upload_check()
        elif path == "/api/trends":
            self._trends()
        elif path == "/api/analyze":
            self._sse(AGENT.analyze_stream)
        elif path == "/api/replicate":
            self._sse(AGENT.replicate_stream)
        elif path == "/api/edit":
            self._edit()
        elif path == "/api/agent_edit":
            self._agent_edit()
        else:
            self.send_error(404)

    def _payload(self):
        try:
            return json.loads(self._read_body() or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _find_by_digest(self, digest: str):
        """按 <digest>_ 前缀在 uploads 目录里查已有文件，返回文件名或 None。"""
        if not digest:
            return None
        prefix = digest + "_"
        for existing in os.listdir(UPLOAD_DIR):
            if existing.startswith(prefix) or existing.startswith(digest + "."):
                return existing
        return None

    def _upload_check(self):
        """前端 md5 预检：命中就直接返回 video_uri，不需要真上传。"""
        payload = self._payload()
        digest = str(payload.get("digest", "")).strip().lower()[:32]
        if not digest:
            self._json({"error": "digest missing"}, 400)
            return
        existing = self._find_by_digest(digest)
        if existing:
            self._json({"exists": True, "video_uri": f"uploads/{existing}", "filename": existing, "digest": digest})
        else:
            self._json({"exists": False, "digest": digest})

    def _upload(self):
        content_type = self.headers.get("Content-Type", "")
        marker = "boundary="
        if marker not in content_type:
            self._json({"error": "multipart/form-data boundary missing"}, 400)
            return
        boundary = content_type.split(marker, 1)[1].strip().strip('"').encode()
        body = self._read_body()
        found = None
        client_digest = ""
        for part in body.split(b"--" + boundary):
            if b"\r\n\r\n" not in part:
                continue
            headers, data = part.split(b"\r\n\r\n", 1)
            head_text = headers.decode("utf-8", "replace")
            if 'name="digest"' in head_text and 'filename=' not in head_text:
                client_digest = data.rstrip(b"\r\n").decode("utf-8", "replace").strip().lower()[:32]
                continue
            if b"filename=" not in part:
                continue
            filename_bits = headers.split(b"filename=", 1)[1]
            raw_name = filename_bits.split(b"\r\n", 1)[0].strip().strip(b'"')
            filename = os.path.basename(raw_name.decode("utf-8", "replace")) or "upload.bin"
            found = (filename, data.rstrip(b"\r\n"))
        if found is None:
            self._json({"error": "file field missing"}, 400)
            return
        filename, data = found
        server_digest = hashlib.sha256(data).hexdigest()[:16]
        # 校验：若前端提供了 digest 但与服务端算出的不一致，仍以服务端为准，避免污染 dedup。
        digest = server_digest
        _, ext = os.path.splitext(filename)
        ext = ext.lower() if ext else ""
        target_name = self._find_by_digest(digest)
        reused = target_name is not None
        if not reused:
            target_name = f"{digest}_{filename}" if filename else f"{digest}{ext or '.bin'}"
            with open(os.path.join(UPLOAD_DIR, target_name), "wb") as stream:
                stream.write(data)
        self._json({
            "video_uri": f"uploads/{target_name}",
            "filename": filename,
            "size": len(data),
            "digest": digest,
            "client_digest_matched": bool(client_digest) and client_digest == digest,
            "reused": reused,
        })
        _log.info("upload %s size=%d reused=%s digest=%s", filename, len(data), reused, digest)

    def _trends(self):
        payload = self._payload()
        skill = skill_mod.get(payload.get("skill_id", ""))
        hint = skill.prompt_hint if skill else ""
        try:
            items = asyncio.run(trends.fetch_trends(payload.get("industry_id", "ecom"), payload.get("intent", ""), hint))
            self._json({"trends": items})
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": str(exc), "trends": []}, 500)

    def _sse(self, make_gen):
        payload = self._payload()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(event):
            self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            bundle = load(payload)

            async def drain():
                async for event in make_gen(bundle):
                    emit(event)

            asyncio.run(drain())
        except Exception as exc:
            _log.error("SSE stream failed: %s", exc, exc_info=True)
            traceback.print_exc()
            try:
                emit({"type": "error", "message": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _edit(self):
        """把 Agent 编排脚本接到 Split 编排层之后，流式跑真实剪辑/TTS/字幕/BGM。"""
        payload = self._payload()
        strategy_path = str(payload.get("strategy_path", "")).strip()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(event):
            self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            if not strategy_path:
                emit({"type": "error", "message": "缺少 strategy_path"})
            else:
                for event in connector.run_edit(
                    strategy_path,
                    enable_tts=bool(payload.get("enable_tts", True)),
                    enable_bgm=bool(payload.get("enable_bgm", True)),
                    enable_t2v=bool(payload.get("enable_t2v", False)),
                    bgm_path=str(payload.get("bgm_path", "")).strip(),
                ):
                    emit(event)
        except Exception as exc:
            _log.error("edit stream failed: %s", exc, exc_info=True)
            traceback.print_exc()
            try:
                emit({"type": "error", "message": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _agent_edit(self):
        """纯 Agent 剪辑链路（agent_cut）：剪辑 Agent + 审片 Agent 重剪循环，流式回传。"""
        payload = self._payload()
        strategy_path = str(payload.get("strategy_path", "")).strip()
        enable_bgm = bool(payload.get("enable_bgm", True))
        max_loops = payload.get("max_loops")
        review_model = str(payload.get("review_model", "qwen")).strip() or "qwen"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(event):
            self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            async def drain():
                if not strategy_path:
                    emit({"type": "error", "message": "缺少 strategy_path"})
                    return
                async for event in agent_edit_loop.agent_edit_stream(
                        strategy_path, enable_bgm=enable_bgm,
                        max_loops=int(max_loops) if max_loops else None,
                        review_model=review_model):
                    emit(event)
            asyncio.run(drain())
        except Exception as exc:
            _log.error("agent_edit stream failed: %s", exc, exc_info=True)
            traceback.print_exc()
            try:
                emit({"type": "error", "message": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        _log.info("%s %s", self.address_string(), fmt % args)


def main():
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), Handler)
    shown = host if host != "0.0.0.0" else "0.0.0.0（本机所有网卡）"
    print(f"Viral Video Agent running at http://{shown}:{port}")
    _log.info("server listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
