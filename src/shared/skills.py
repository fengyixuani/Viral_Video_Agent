"""Thin AgentScope SKILL.md loading adapter."""
import asyncio
import os
from dataclasses import dataclass, field

import frontmatter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKILL_DIR = os.path.join(ROOT, "skills_defs")


@dataclass
class Skill:
    id: str
    name: str
    icon: str
    desc: str
    industry: str
    scheme_hint: str = "faithful"
    operator_pipeline: list = field(default_factory=list)
    preset_dims: list = field(default_factory=list)
    prompt_hint: str = ""
    hidden: bool = False

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "icon": self.icon,
            "desc": self.desc, "industry": self.industry,
            "scheme_hint": self.scheme_hint, "preset_dims": self.preset_dims,
            "prompt_hint": self.prompt_hint,
        }


_SKILLS = {}


def _directories():
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("use filesystem fallback inside an active event loop")
        from agentscope.skill import LocalSkillLoader
        loader = LocalSkillLoader(directory=SKILL_DIR, scan_subdir=True)
        listed = asyncio.run(loader.list_skills())
        return [os.path.basename(item.dir.rstrip("/")) for item in listed]
    except Exception:
        if not os.path.isdir(SKILL_DIR):
            return []
        return [name for name in os.listdir(SKILL_DIR)
                if os.path.isfile(os.path.join(SKILL_DIR, name, "SKILL.md"))]


def reload():
    """Reload all local AgentScope skill definitions."""
    global _SKILLS
    loaded = {}
    for dirname in sorted(set(_directories())):
        path = os.path.join(SKILL_DIR, dirname, "SKILL.md")
        try:
            post = frontmatter.load(path)
            meta = post.metadata
            skill_id = str(meta.get("skill_id", dirname))
            loaded[skill_id] = Skill(
                id=skill_id,
                name=str(meta.get("name", skill_id)),
                icon=str(meta.get("icon", "Skill")),
                desc=str(meta.get("description", "")),
                industry=str(meta.get("industry", "ecom")),
                scheme_hint=str(meta.get("scheme_hint", "faithful")),
                operator_pipeline=list(meta.get("operator_pipeline", []) or []),
                preset_dims=list(meta.get("preset_dims", []) or []),
                prompt_hint=post.content.strip(),
                hidden=bool(meta.get("hidden", False)),
            )
        except (OSError, TypeError, ValueError):
            continue
    _SKILLS = loaded
    return list(_SKILLS.values())


def get(skill_id):
    if not _SKILLS:
        reload()
    return _SKILLS.get(skill_id)


def skill_list():
    """前端推荐栏：仅返回非隐藏的场景 skill。"""
    if not _SKILLS:
        reload()
    return [skill.to_dict() for skill in _SKILLS.values() if not skill.hidden]
