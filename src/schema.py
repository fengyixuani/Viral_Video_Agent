"""Core business data contracts."""
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ShotSlot:
    id: int
    want: str
    duration: float
    role: str = ""
    breakdown: list = field(default_factory=list)
    narrative: dict = field(default_factory=dict)
    camera: dict = field(default_factory=dict)
    visual: dict = field(default_factory=dict)
    audio: dict = field(default_factory=dict)
    thumb: str = ""
    remake_thumb: str = ""


@dataclass
class BaseTemplate:
    template_id: str
    industry: str
    total_duration_sec: float
    hook: dict
    narrative_structure: list[str]
    rhythm: dict
    shot_slots: list[ShotSlot]
    selling_points_order: list[str] = field(default_factory=list)
    cta: dict = field(default_factory=dict)
    packaging: dict = field(default_factory=dict)
    dropped_trailing_slots: dict = field(default_factory=dict)


@dataclass
class MaterialClip:
    clip_id: str
    uri: str
    tags: list[str]
    fillable_slots: list[str]
    quality: Literal["A", "B", "C"]
    duration: float


@dataclass
class MaterialProfile:
    material_id: str
    clips: list[MaterialClip]
    capability: dict = field(default_factory=dict)


@dataclass
class SlotAssignment:
    slot_id: int
    action: Literal["match", "generate"]
    material_clip_id: Optional[str] = None
    gen_prompt: Optional[str] = None


@dataclass
class ReplicationPlan:
    plan_id: str
    granularity: Literal["full", "action_scene", "style"]
    confidence: float
    source: Literal["from-default", "clarified", "user-edited"]
    slot_assignments: list[SlotAssignment]
    packaging: dict = field(default_factory=dict)
    voice_strategy: dict = field(default_factory=dict)
    material_strategy: Literal["faithful", "balanced", "regenerate"] = "faithful"


@dataclass
class VideoFile:
    uri: str
    duration_sec: float
    shots: list[dict] = field(default_factory=list)
