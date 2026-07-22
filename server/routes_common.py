"""公共/基础设施路由：技能列表、项目列表、素材上传、趋势。三人共用。"""
import asyncio
import hashlib
import os
import traceback

from _paths import UPLOAD_DIR
import obs
import skills as skill_mod
import projects
import runs
import trends

_log = obs.get_logger("http.common")


def handle_runs(h):
    """运行记录：列出每次运行（编排/剪辑/短剧）的可读条目，最新在前。"""
    h._json({"runs": runs.list_runs()})


def handle_skills(h):
    h._json({"skills": skill_mod.skill_list()})


def handle_projects(h):
    h._json({"projects": projects.list_projects()})


def handle_trends(h):
    payload = h._payload()
    skill = skill_mod.get(payload.get("skill_id", ""))
    hint = skill.prompt_hint if skill else ""
    try:
        items = asyncio.run(trends.fetch_trends(payload.get("industry_id", "ecom"),
                                                payload.get("intent", ""), hint))
        h._json({"trends": items})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        h._json({"error": str(exc), "trends": []}, 500)


def handle_upload_check(h):
    """前端 md5 预检：命中就直接返回 video_uri，不需要真上传。"""
    payload = h._payload()
    digest = str(payload.get("digest", "")).strip().lower()[:32]
    if not digest:
        h._json({"error": "digest missing"}, 400)
        return
    existing = h._find_by_digest(digest)
    if existing:
        h._json({"exists": True, "video_uri": f"uploads/{existing}", "filename": existing, "digest": digest})
    else:
        h._json({"exists": False, "digest": digest})


def handle_upload(h):
    content_type = h.headers.get("Content-Type", "")
    marker = "boundary="
    if marker not in content_type:
        h._json({"error": "multipart/form-data boundary missing"}, 400)
        return
    boundary = content_type.split(marker, 1)[1].strip().strip('"').encode()
    body = h._read_body()
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
        h._json({"error": "file field missing"}, 400)
        return
    filename, data = found
    digest = hashlib.sha256(data).hexdigest()[:16]
    _, ext = os.path.splitext(filename)
    ext = ext.lower() if ext else ""
    target_name = h._find_by_digest(digest)
    reused = target_name is not None
    if not reused:
        target_name = f"{digest}_{filename}" if filename else f"{digest}{ext or '.bin'}"
        with open(os.path.join(UPLOAD_DIR, target_name), "wb") as stream:
            stream.write(data)
    h._json({
        "video_uri": f"uploads/{target_name}",
        "filename": filename,
        "size": len(data),
        "digest": digest,
        "client_digest_matched": bool(client_digest) and client_digest == digest,
        "reused": reused,
    })
    _log.info("upload %s size=%d reused=%s digest=%s", filename, len(data), reused, digest)


GET = {
    "/api/skills": handle_skills,
    "/api/projects": handle_projects,
    "/api/runs": handle_runs,
}
POST = {
    "/api/upload": handle_upload,
    "/api/upload/check": handle_upload_check,
    "/api/trends": handle_trends,
}
