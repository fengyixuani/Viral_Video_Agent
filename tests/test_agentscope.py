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
import react_agents


class AgentScopeTests(unittest.TestCase):
    def test_toolkits_register_expected_functions(self):
        understanding = react_agents.build_understanding_toolkit()
        planning = react_agents.build_planning_toolkit()
        u_schemas = asyncio.run(understanding.get_tool_schemas())
        p_schemas = asyncio.run(planning.get_tool_schemas())
        u_names = {s["function"]["name"] for s in u_schemas}
        p_names = {s["function"]["name"] for s in p_schemas}
        self.assertIn("parse_reference", u_names)
        self.assertIn("assess_materials", u_names)
        self.assertTrue({"plan_execute", "decide_shot", "generate_shot", "edit_timeline", "package_video"} <= p_names)

    def test_function_tool_schema_from_signature(self):
        toolkit = react_agents.build_understanding_toolkit()
        schemas = asyncio.run(toolkit.get_tool_schemas())
        parse = next(s for s in schemas if s["function"]["name"] == "parse_reference")
        props = parse["function"]["parameters"]["properties"]
        self.assertIn("video_uri", props)
        self.assertIn("duration_sec", props)

    def test_as_core_stream_mock_emits_reasoning_and_content(self):
        async def collect():
            events = []
            async for item in as_core.stream("你是拆解 Agent", "{\"video_desc\":\"test\"}", vision=False):
                events.append(item)
            return events
        events = asyncio.run(collect())
        self.assertTrue(any("reasoning" in e for e in events))
        final = next(e for e in events if "content" in e)
        payload = as_core.parse_json(final["content"])
        self.assertIn("shot_slots", payload)


if __name__ == "__main__":
    unittest.main()
