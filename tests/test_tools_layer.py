"""业务层（src/tools/）核心行为断言。"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tools import (
    EditingTool,
    GenerationTool,
    PackagingTool,
    PlanningTool,
    UnderstandingTool,
)


class ToolsLayerTests(unittest.TestCase):
    def test_understanding_build_template_shapes_shot_slots(self):
        data = {
            "template_id": "tpl-x",
            "total_duration_sec": 12.5,
            "hook": {"text": "开场"},
            "narrative_structure": ["起", "承"],
            "rhythm": {"bpm": 120},
            "shot_slots": [
                {"id": 1, "want": "露出主体", "duration": 3.5, "role": "hook", "breakdown": [{"dim": "视觉", "value": "特写"}]},
                {"id": 2, "want": "痛点", "duration": 4.0, "role": "body"},
            ],
            "selling_points_order": ["点1"],
            "cta": {"text": "点击"},
        }
        template = UnderstandingTool().build_template(data, "ecom")
        self.assertEqual(template.industry, "ecom")
        self.assertEqual(len(template.shot_slots), 2)
        self.assertEqual(template.shot_slots[0].want, "露出主体")
        self.assertAlmostEqual(template.shot_slots[0].duration, 3.5)
        self.assertEqual(template.total_duration_sec, 12.5)
        self.assertTrue(template.shot_slots[0].breakdown)

    def test_generation_run_returns_mock_uri_and_prompt(self):
        result = GenerationTool().run(gen_prompt="一个特写镜头", duration=2.5)
        self.assertEqual(result["uri"], "mock://generated/placeholder.mp4")
        self.assertEqual(result["prompt"], "一个特写镜头")
        self.assertAlmostEqual(result["duration"], 2.5)
        self.assertEqual(result["engine"], "seedance-mock")

    def test_planning_payloads_are_structured(self):
        planning = PlanningTool()
        plan = planning.plan_payload("方案A", "faithful", ["d1"], ["t1"], 3)
        self.assertEqual(plan, {
            "scheme_name": "方案A",
            "strategy": "faithful",
            "dimensions": ["d1"],
            "trends": ["t1"],
            "materials_count": 3,
        })
        decide = planning.decide_payload(2, "痛点", "balanced")
        self.assertEqual(decide, {"slot_id": 2, "want": "痛点", "strategy": "balanced"})

    def test_editing_and_packaging_use_new_signatures(self):
        edited = EditingTool().run(shots=[{"slot_id": 1}])
        self.assertEqual(edited["shots"], [{"slot_id": 1}])
        video = PackagingTool().run(shots=[{"slot_id": 1}], duration_sec=7.0)
        self.assertEqual(video.uri, "mock://output/final.mp4")
        self.assertAlmostEqual(video.duration_sec, 7.0)
        self.assertEqual(video.shots, [{"slot_id": 1}])


if __name__ == "__main__":
    unittest.main()
