"""B 负责：编排 的路由（/api/replicate）。

消费 A 的 AnalysisResult（契约①，随请求体传入）→ 产出 strategy JSON（契约②）。
剪辑相关路由见 routes_editing.py（同为 B）。
"""
import obs
from orchestration import OrchestrationAgent

_log = obs.get_logger("http.orchestration")
_AGENT = OrchestrationAgent()


def handle_replicate(h):
    h._sse(_AGENT.replicate_stream)


GET = {}
POST = {
    "/api/replicate": handle_replicate,
}
