"""模块间数据契约的单一事实来源（Single Source of Truth）。

定义三段流水线交接的 JSON 结构约束 + 轻量校验函数（返回问题列表，空列表=通过）。
校验是**非致命**的：调用方拿到 problems 后记 warning，不打断流程——目的是当"生产端写的
字段"与"消费端读的字段"不一致时**立刻被发现**（尤其 A→B 的 EditingStrategy）。

覆盖的契约（详见 docs/contracts.md）：
- InputBundle：已是 dataclass，见 shared/inputs/loader.py，本文件不重复。
- AnalysisResult（契约①，A 内部）：understanding/orchestrator.analyze_stream 产出。
- EditingStrategy（契约②，A→B 最关键）：``*_selected_editing_strategy_*.json``。
- ConnectorContext：``connector_context.json``（{shots, segments}）。

改契约字段时：改 CONTRACT_VERSION、更新本文件校验、同步 docs/contracts.md。
"""
from __future__ import annotations

# 契约版本：任何字段级不兼容变更都应 +1，并同步 docs/contracts.md
CONTRACT_VERSION = "1.0"

# --------------------------------------------------------------------------- #
# 契约②：EditingStrategy（A 产出 / B 消费，最关键）
# 顶层：metadata / user_asset_bank / slot_matching / editing_timeline /
#       missing_assets / overall_editing_strategy
# --------------------------------------------------------------------------- #
STRATEGY_REQUIRED_TOP = ("metadata", "editing_timeline")
TIMELINE_ITEM_REQUIRED = ("slot_id", "action")
TIMELINE_ACTIONS = ("use_user_asset", "generate", "reshoot")


def validate_strategy(strategy) -> list:
    """校验 EditingStrategy（strategy JSON）。返回问题列表（空=通过）。"""
    problems: list = []
    if not isinstance(strategy, dict):
        return ["strategy 不是 dict"]
    for k in STRATEGY_REQUIRED_TOP:
        if k not in strategy:
            problems.append(f"缺少顶层字段 `{k}`")
    tl = strategy.get("editing_timeline")
    if not isinstance(tl, list) or not tl:
        problems.append("editing_timeline 应为非空列表")
    else:
        for i, it in enumerate(tl):
            if not isinstance(it, dict):
                problems.append(f"editing_timeline[{i}] 不是 dict")
                continue
            for k in TIMELINE_ITEM_REQUIRED:
                if not it.get(k):
                    problems.append(f"editing_timeline[{i}] 缺少 `{k}`")
            act = it.get("action")
            if act and act not in TIMELINE_ACTIONS:
                problems.append(f"editing_timeline[{i}].action=`{act}` 非法（应 ∈ {TIMELINE_ACTIONS}）")
            if act == "use_user_asset" and not it.get("source_path"):
                problems.append(f"editing_timeline[{i}] action=use_user_asset 但缺 source_path（B 无从剪辑）")
    ma = strategy.get("missing_assets")
    if ma is not None and not isinstance(ma, list):
        problems.append("missing_assets 若存在应为列表")
    bank = strategy.get("user_asset_bank")
    if bank is not None and not isinstance(bank, list):
        problems.append("user_asset_bank 若存在应为列表")
    return problems


# --------------------------------------------------------------------------- #
# ConnectorContext：connector_context.json（B 全池召回 / AIGC 补镜用）
# --------------------------------------------------------------------------- #
def validate_connector_context(ctx) -> list:
    """校验 connector_context（{shots, segments}）。返回问题列表（空=通过）。"""
    problems: list = []
    if not isinstance(ctx, dict):
        return ["connector_context 不是 dict"]
    if not isinstance(ctx.get("shots"), list):
        problems.append("shots 应为列表")
    segs = ctx.get("segments")
    if not isinstance(segs, list):
        problems.append("segments 应为列表")
    else:
        for i, s in enumerate(segs[:500]):
            if isinstance(s, dict) and not s.get("source_path"):
                problems.append(f"segments[{i}] 缺 source_path")
    return problems


# --------------------------------------------------------------------------- #
# 契约①：AnalysisResult（A 内部，理解→编排；较宽松）
# --------------------------------------------------------------------------- #
def validate_analysis_result(res) -> list:
    """校验 analyze 产出的 result（template / schemes / …）。返回问题列表（空=通过）。"""
    problems: list = []
    if not isinstance(res, dict):
        return ["analysis result 不是 dict"]
    tpl = res.get("template")
    if not isinstance(tpl, dict):
        problems.append("template 应为 dict")
    elif not isinstance(tpl.get("shot_slots"), list):
        problems.append("template.shot_slots 应为列表")
    if not isinstance(res.get("schemes"), list):
        problems.append("schemes 应为列表")
    return problems


def summarize(problems) -> str:
    """把问题列表压成一行日志文本。"""
    return "OK" if not problems else f"{len(problems)} 处不符契约：" + "；".join(problems[:8])
