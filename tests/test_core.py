import asyncio
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for _p in (os.path.join(SRC, "shared"), SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import as_core
import skills
from understanding import UnderstandingAgent
from orchestration import OrchestrationAgent
from inputs import load
from trends import fetch_trends


class CoreTests(unittest.TestCase):
    def test_parse_json_ignores_fence_and_suffix(self):
        self.assertEqual(as_core.parse_json('```json\n{"ok": 1}\n``` trailing'), {"ok": 1})

    def test_default_model_routing(self):
        self.assertEqual(as_core.pick_model(vision=True), "ali-qwen3.7-plus")
        self.assertEqual(as_core.pick_model(vision=False), "ali-qwen3.7-max")

    def test_skills_load_five_definitions(self):
        loaded = skills.reload()
        visible_ids = {s["id"] for s in skills.skill_list()}
        # 隐藏 skill 不进推荐栏
        self.assertNotIn("user_material_understanding", visible_ids)
        self.assertNotIn("viral_reference_understanding", visible_ids)
        # 通用理解基础层 + 用户素材理解均为隐藏 skill
        self.assertTrue(skills.get("user_material_understanding").hidden)
        self.assertTrue(skills.get("viral_reference_understanding").hidden)
        # 电商理解增强作为可叠加的场景 skill 出现在推荐栏
        self.assertIn("ecom_understanding", visible_ids)
        self.assertEqual(skills.get("drama").industry, "drama")
        self.assertTrue(all(item.prompt_hint for item in loaded))

    def test_input_normalization(self):
        bundle = load({"intent": " x ", "materials": ["a"], "template": []})
        self.assertEqual(bundle.intent, "x")
        self.assertEqual(bundle.materials, ["a"])
        self.assertEqual(bundle.template, {})

    def test_mock_trends_are_normalized(self):
        result = asyncio.run(fetch_trends("ecom", "测试"))
        self.assertGreaterEqual(len(result), 8)
        self.assertTrue(all(item["keyword"] for item in result))

    def test_analyze_and_replicate_streams(self):
        understanding_agent = UnderstandingAgent()
        orchestration_agent = OrchestrationAgent()

        async def collect(gen):
            return [item async for item in gen]

        analyze = asyncio.run(collect(understanding_agent.analyze_stream(load({"skill_id": "drama"}))))
        result = next(item["result"] for item in analyze if item["type"] == "analysis")
        self.assertEqual(result["industry_guess"], "drama")
        self.assertGreaterEqual(len(result["schemes"]), 2)
        self.assertTrue(result["template"]["shot_slots"][0]["breakdown"])
        replicate = asyncio.run(collect(orchestration_agent.replicate_stream(load({
            "industry_id": "drama", "template": result["template"],
            "selected_dimensions": ["fine-hook"], "material_strategy": "balanced",
        }))))
        final = next(item["result"] for item in replicate if item["type"] == "final")
        self.assertTrue(final["plan"]["slot_assignments"])
        self.assertEqual(final["video"]["uri"], "mock://output/final.mp4")


if __name__ == "__main__":
    unittest.main()
