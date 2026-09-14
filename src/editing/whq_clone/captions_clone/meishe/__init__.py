"""美摄花字后端: 醒目块套美摄【花字 styleId】、口播套美摄【字体】, 云端 RenderVideo 出片。

移植自 copy_zimu/v2/meishe/*。分工:
  - catalog.py            读 assets/ 下的花字/字体资产快照(随包携带, 无外部依赖)
  - fancy_match.py        LLM 给每块挑花字/字体(走 Agent 网关)
  - render_meishe_job.py  独立脚本: blocks.json -> 美摄统一 schema -> 提交 RenderVideo -> 下载成片
                          (必须在 meishe repo 环境/python3.13 下跑, 由 render.py 起子进程)
  - render.py             编排上面三步, 对外只暴露 burn()

美摄服务与 BOS SDK 只在 meishe repo 环境里可用, 所以渲染一步走子进程(同 finisher 迁移 BGM
的既有做法), 不可达时优雅返回 None, 由调用方回退本机 ASS 烧录。
"""
from .render import burn  # noqa: F401

__all__ = ["burn"]
