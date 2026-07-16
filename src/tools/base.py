"""Tool protocol and mock base class."""
from typing import Any, Protocol


class Tool(Protocol):
    name: str
    def run(self, payload: Any, profile: Any) -> Any: ...


class MockTool:
    name = "tool"

    def _log(self, msg: str):
        line = f"[MOCK tool:{self.name}] {msg}"
        print(line)
        return line
