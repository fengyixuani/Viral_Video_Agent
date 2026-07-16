"""Normalize HTTP payloads into the agent input contract."""
from dataclasses import dataclass, field


@dataclass
class InputBundle:
    video_uri: str = ""
    video_desc: str = ""
    intent: str = ""
    skill_id: str = ""
    industry_id: str = "ecom"
    template: dict = field(default_factory=dict)
    selected_dimensions: list[str] = field(default_factory=list)
    selected_trends: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    material_strategy: str = "faithful"
    scheme_name: str = ""
    enabled_tools: list[str] | None = None  # None = 不限（全部工具可用）
    use_cache: bool = True
    refresh_vectors: bool = False
    reproduce_mode: str = "structure"  # structure=结构优先 / shot=镜头优先


def _string(payload, key, default=""):
    value = payload.get(key, default)
    return value if isinstance(value, str) else default


def _list(payload, key):
    value = payload.get(key, [])
    return value if isinstance(value, list) else []


def load(payload) -> InputBundle:
    payload = payload if isinstance(payload, dict) else {}
    template = payload.get("template", {})
    enabled = payload.get("enabled_tools")
    use_cache = payload.get("use_cache", True)
    return InputBundle(
        video_uri=_string(payload, "video_uri"),
        video_desc=_string(payload, "video_desc").strip(),
        intent=_string(payload, "intent").strip(),
        skill_id=_string(payload, "skill_id"),
        industry_id=_string(payload, "industry_id", "ecom") or "ecom",
        template=template if isinstance(template, dict) else {},
        selected_dimensions=_list(payload, "selected_dimensions"),
        selected_trends=_list(payload, "selected_trends"),
        materials=_list(payload, "materials"),
        material_strategy=_string(payload, "material_strategy", "faithful") or "faithful",
        scheme_name=_string(payload, "scheme_name"),
        enabled_tools=[str(x) for x in enabled] if isinstance(enabled, list) else None,
        use_cache=bool(use_cache),
        refresh_vectors=bool(payload.get("refresh_vectors", False)),
        reproduce_mode=_string(payload, "reproduce_mode", "structure") or "structure",
    )
