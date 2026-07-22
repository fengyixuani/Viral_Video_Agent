"""Data-driven industry profiles."""
from dataclasses import dataclass, field


@dataclass
class Profile:
    industry: str
    label: str = ""
    dimension_weights: dict = field(default_factory=dict)
    toolkit: dict = field(default_factory=dict)
    judge_rubric: dict = field(default_factory=dict)
    workflows: list = field(default_factory=list)


ECOM = Profile("ecom", "电商营销", {"hook": 1.2, "product": 1.2, "cta": 1.0}, {"tts": True}, {"conversion": 1.0}, ["storyboard"])
LIVE = Profile("live", "直播带货切片", {"hook": 1.2, "emotion": 1.2}, {"tts_clone": True}, {"highlight": 1.0}, ["highlight_clip"])
DRAMA = Profile("drama", "短剧", {"conflict": 1.2, "reversal": 1.2}, {"video_generation": True}, {"story": 1.0}, ["story"])
KNOWLEDGE = Profile("knowledge", "知识科普", {"clarity": 1.2, "density": 1.0}, {"graphics": True}, {"accuracy": 1.0}, ["explain"])
REGISTRY = {p.industry: p for p in (ECOM, LIVE, DRAMA, KNOWLEDGE)}
INDUSTRY_OPTIONS = [{"id": p.industry, "label": p.label} for p in REGISTRY.values()]


def load_profile(industry_id: str) -> Profile:
    return REGISTRY.get(industry_id, ECOM)
