"""Prompt construction and template conversion for understanding."""
import json

from schema import BaseTemplate, MaterialProfile, ShotSlot


class UnderstandingTool:
    name = "解析"

    def schema_hint(self):
        return '''{"industry_guess":"ecom|live|drama|knowledge","industry_reason":"str","total_duration_sec":0,"hook":{},"narrative_structure":["str"],"rhythm":{},"selling_points_order":["str"],"cta":{},"shot_slots":[{"id":1,"source_time_range":"0.00-3.50","want":"str","duration":3.0,"role":"str","breakdown":[{"dim":"str","value":"str"}]}],"schemes":[{"id":"str","name":"str","strategy":"faithful|balanced|regenerate","desc":"str","dimensions":[{"id":"str","name":"str","level":"coarse|fine","desc":"str","recommended":true,"replace":{"enabled":true,"types":["image|text|video"],"hint":"str"}}]}]}'''

    def analyze_messages(self, video_uri, video_desc, intent, materials, duration_sec=None, base_prompt=None, overlays=None):
        """组合式理解 prompt：通用理解 skill 作为基础层，可叠加场景 skill。

        Args:
            base_prompt: 通用理解 skill 正文；为空时回退内置默认。
            overlays: 叠加层列表，每项 ``(skill_name, prompt)``，按序追加到 system。
        """
        base = base_prompt or (
            "你是通用爆款视频拆解专家。推断 ecom/live/drama/knowledge 行业；"
            "请自行观察参考视频，按真实剪辑节奏逐镜拆解，不要事先限定镜头数量；"
            "每个 shot_slot 与视频里的一个真实分镜对齐，duration 与实际镜头时长一致，所有 shot 的 duration 之和应约等于 total_duration_sec，覆盖完整视频。"
            "每镜给出 5 到 9 个由行业和作用决定的可变 breakdown 维度；维度名与方案 dimensions 呼应。"
            "生成 2 到 3 组自主命名方案，strategy 为 faithful/balanced/regenerate，维度从 coarse 到 fine 并判断是否需补素材。"
            "所有字段值必须使用简体中文（role 用中文如 开场钩子/产品登场/使用演示/卖点强化/行动号召，禁止 hook_visual、product_close_up 这类英文标识），"
            "仅 industry、strategy、level 枚举保留规定英文取值，其余一律中文，不要输出英文单词或拼音。"
        )
        parts = [base]
        for name, prompt in (overlays or []):
            if prompt:
                parts.append(f"\n\n【叠加 skill：{name}】\n{prompt}")
        # 硬性要求（不随 skill 变）：每个 shot_slot 必须给出它在参考视频里的**真实起止时间** source_time_range，
        # 格式 "起秒-止秒"（如 "0.00-3.50"），与该镜的画面/文字描述严格对应；相邻镜头首尾相接、覆盖整段视频、
        # 不重叠不留空。下游据此抽该镜缩略图，务必与描述同源、别用估算。
        parts.append(
            "\n\n【逐镜时间轴（必须遵守）】\n"
            "- 你会以约每秒数帧的密度看到画面。请**逐帧观察画面在什么时刻发生真实切换**（机位/主体/景别突变即为一次镜头切点），"
            "以真实切点划分镜头，**不要把时间平均等分**、不要用整齐的 1 秒一镜这种估算。\n"
            "- 每个 shot_slot 必须输出 source_time_range = \"起秒-止秒\"（保留两位小数，如 \"2.35-3.80\"），"
            "即该镜真实起止时间，且必须与该镜的画面内容/breakdown 描述严格对应——描述的是哪一段画面，时间戳就落在哪一段。\n"
            "- 相邻镜头首尾相接（前一镜的止 = 后一镜的起），覆盖 0 到 total_duration_sec 整段，不重叠、不留空。\n"
            "- duration 应等于 source_time_range 的时长。若某几镜明显长/短，如实给出不同时长，不要凑成一样。"
        )
        system = "\n".join(parts) + "\n严格只输出 JSON：" + self.schema_hint()
        user = json.dumps({
            "video_uri": video_uri,
            "video_desc": video_desc,
            "intent": intent,
            "materials": materials,
            "duration_sec": duration_sec or 0,
        }, ensure_ascii=False)
        return system, user

    @staticmethod
    def _compat_breakdown(item):
        result = []
        labels = (("narrative", "叙事"), ("camera", "镜头语言"), ("visual", "视觉"), ("audio", "声音"))
        for key, label in labels:
            value = item.get(key)
            if value:
                text = ", ".join(f"{k}:{v}" for k, v in value.items()) if isinstance(value, dict) else str(value)
                result.append({"dim": label, "value": text})
        return result

    def build_template(self, data, industry):
        data = data if isinstance(data, dict) else {}
        slots = []
        for index, item in enumerate(data.get("shot_slots", []) or []):
            if not isinstance(item, dict):
                continue
            slots.append(ShotSlot(id=int(item.get("id", index + 1)), want=str(item.get("want", "")), duration=float(item.get("duration", 0.0) or 0.0), role=str(item.get("role", "")), breakdown=item.get("breakdown", []) or self._compat_breakdown(item), narrative=item.get("narrative", {}) or {}, camera=item.get("camera", {}) or {}, visual=item.get("visual", {}) or {}, audio=item.get("audio", {}) or {}, source_time_range=str(item.get("source_time_range", "") or ""), thumb=str(item.get("thumb", "") or ""), remake_thumb=str(item.get("remake_thumb", "") or "")))
        return BaseTemplate(template_id=str(data.get("template_id", "tpl-llm")), industry=industry, total_duration_sec=float(data.get("total_duration_sec", 0.0) or 0.0), hook=data.get("hook", {}) or {}, narrative_structure=data.get("narrative_structure", []) or [], rhythm=data.get("rhythm", {}) or {}, shot_slots=slots, selling_points_order=data.get("selling_points_order", []) or [], cta=data.get("cta", {}) or {}, packaging=data.get("packaging", {}) or {}, dropped_trailing_slots=data.get("dropped_trailing_slots", {}) or {})

    def profile_materials(self, materials):
        return MaterialProfile(material_id="mat-mock", clips=[], capability={"provided_count": len(materials or [])})
