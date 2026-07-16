# 素材召回与入库重设计（Retrieval Redesign）

> 面向 Viral_Video_Agent 的可行性验证（feasibility）与素材匹配场景。
> 目标：解决当前实现的 4 类问题（污染、噪声、单点、鲁棒性），并在参考 Viral_Video_Split 的基础上做出更适合 Agent 交互形态的选择。

## 0. TL;DR — 一句话结论

- **入库**：per-task 隔离的持久化 JSON，key 用「摘要 + 视觉描述」二段拼接，keywords 和口播不进 key。
- **检索**：多 query（L1 意图 + L2 视觉 + 可选 L3 LLM 改写）→ 每 query 独立余弦排序 → **RRF 融合** → top-k 交给 LLM 二判。
- **阈值**：去掉 embedding 侧硬阈值 0.62/0.45，只保留「所有 query 全空则 none」的最终兜底，判定权全部让给 LLM 打分。
- **鲁棒性**：千帆 API 失败自动降级到本地 BM25/关键词 lexical fallback，pipeline 不断链。
- **接口**：合并 `RetrievalTool.retrieve` 与 `VectorStore.search` 为单一 `Retriever` 门面，query embedding 内存缓存。

---

## 1. 当前实现的问题清单

（详见 `src/tools/retrieval.py` / `src/tools/vectorstore.py` / `src/agent/feasibility.py`）

| # | 问题 | 影响 |
|---|---|---|
| P0 | `store.json` 全局单文件累积，跨 task/user 混在一起（日志显示已累计 100+） | 新素材被历史素材淹没，召回错位 |
| P1 | Key 拼接了 `keywords` + `speech_or_text`（ASR 口播） | ASR 噪声与语气词拉偏视觉功能匹配 |
| P1 | 单 query（`role + want + 过滤后 dim_text`），dim 白名单过严导致常退化 | 表达力弱、多样性不足 |
| P2 | Embedding 侧硬阈值 0.62 / 0.45 | 模型/域漂移时误判率高 |
| P2 | 千帆 API 挂 = pipeline 断链，无 fallback | 稳定性差 |
| P3 | `_segment_text` 在两个文件里各写一份 | 维护同步风险 |
| P3 | 没有 embedding 缓存 | 重复 embed 浪费网络与费用 |
| P4 | Query 和 matches 不落 log | 无法回溯 bad case |
| P4 | SKILL.md 承诺「后处理补 `global_asset_id`」但 `material_understanding.py` 未实现，靠 `vectorstore.py:89` 兜底 | 跨 source 入库时 id 不稳，去重与回查失效（详见 §11.2.2） |
| P4 | SKILL.md `入库要求` 显式点名 `keywords / speech_or_text 也入库`，与新 key 策略矛盾 | 数据契约层面需要同步（详见 §11.2.1） |
| P4 | VLM 已产出的 `asset_type / visual_evidence_tags / quality_score / actions / visible_objects` 全被入库丢弃 | 浪费上游信号，LLM 二判缺关键证据（详见 §11.3） |

---

## 2. 设计原则

1. **信号密度 > 信号广度**：宁可用少量高质量字段做 key，也不把噪声全塞进来。视觉功能匹配不需要口播语气词。
2. **多样性 > 单点 top-1**：真实 case 里，一个 shot 通常需要「主体 + 视角 + 动作」多维度信号，单 query 只覆盖一个维度。
3. **判定权集中在 LLM，不是相似度**：余弦分数在不同 batch/模型/域会漂移，硬阈值不稳。LLM 二判分数由模型显式生成，尺度稳定。
4. **持久化 per-task，不全局累积**：Agent 是交互式反复重跑场景，per-task 缓存能省 embed 费用；但跨任务必须完全隔离。
5. **降级链要完整**：千帆 API 挂 → 本地 lexical → 至少给 LLM 一批候选。绝不空手让 pipeline 断链。
6. **可观测性内置**：query、top matches、scores 必须能被日志或 dump 抓到。

---

## 3. 入库阶段（Indexing）

### 3.1 存储布局

**路径改为 per-task 目录**：

```
uploads/vectors/{task_id}/store.json
```

`task_id` 从上下文取（`analyze_id` / `session_id` / 前端传入的 task 标识）。设计动机：
- 交互式 Agent 允许反复重跑分析，per-task 持久化能省 embed 费用；
- 跨任务的素材语义空间完全隔离，避免污染；
- 调试时可直接删单个目录清空。

**记录 schema**（对齐 `user_material_understanding` skill 的实际输出，见 §11）：

```json
{
  "id": "global_asset_id",
  "text": "被 embed 的最终文本",
  "vector": [...],
  "meta": {
    "asset_id": "...",
    "source_video_id": "...",
    "source_time_range": "0.0-1.2",
    "duration": 1.2,                    // 从 source_time_range 解析
    "asset_type": "商品特写",            // VLM 输出的类型标签，用于 pre-filter / boost
    "one_sentence_summary": "...",
    "visual_description": "...",
    "keywords": [...],                  // 从 key 移到 meta，供 lexical fallback
    "actions": [...],                   // VLM 输出，lexical fallback 词表
    "visible_objects": [...],           // VLM 输出，lexical fallback 词表
    "visual_evidence_tags": [...],      // VLM 输出，喂给 LLM 二判的证据标签
    "quality_score": 0.95,              // VLM 自评的可用度，RRF 后 tiebreaker
    "limitations": [...],               // VLM 标注的局限性，观测/调试用
    "indexed_at": 1721030000
  }
}
```

`indexed_at` 用于将来做 TTL / GC；`keywords / actions / visible_objects` 均挪到 meta 便于下游 lexical fallback 使用；`asset_type / visual_evidence_tags / quality_score` 是 VLM 已经产出但当前入库丢弃的字段（见 §11.2）。

### 3.2 Key 拼接策略

**只用 2 段**：

```python
def _segment_text(seg: dict) -> str:
    summary = seg.get("one_sentence_summary") or ""
    visual  = seg.get("visual_description") or ""
    text = f"{summary} {visual}".strip()
    return text or seg.get("asset_id", "")
```

理由：
- `one_sentence_summary` 是 VLM 高浓缩，信号密度最高；
- `visual_description` 补充细粒度视觉细节，对"景别/动作/构图"类 query 有效；
- 排除 `keywords`：词袋噪声，且视觉大模型经常给出重复/无信息量的关键词；
- **排除 `speech_or_text`（口播/ASR）**：这是文本模态，与"视觉功能匹配"目标不符（例如"姐妹们冲、家人们..."会拉偏语义空间）。
- Query 侧本来也不带口播，key 侧带反而导致空间不对齐。

**可选开关**：`RAG_INCLUDE_SPEECH=1` 时把口播也拼入，用于口播强相关场景（如"人物讲解"类 shot）。默认关闭。

### 3.3 去重

以 `global_asset_id`（缺失时用 `{source}::{asset_id}`）为主键，per-task 目录内去重。

### 3.4 Embed 前预处理

- 空文本跳过（当前已有）；
- 单条超过 500 字符截断（视觉描述可能很长，Qwen3-Embedding 长文本收益递减）；
- Batch 上限 32 条一批发千帆，避免单请求过大超时。

---

## 4. 检索阶段（Retrieval）

### 4.1 多 Query 生成（三级）

针对每个 shot，生成 **多条互补 query**，来源分三级：

**L1 — 意图 query**（必生成）
```
"{role}·{want}"
```
例：`"开场钩子·展示产品最佳状态，建立第一印象"`

**L2 — 视觉 query**（有 breakdown 时生成）

从 `breakdown[]` 中挑**视觉/形式类**维度，把值拼成一句自然语言：

- 视觉维度白名单：`景别 / 机位 / 镜头 / 主体 / 展示重点 / 交互动作 / 光影 / 构图 / 素材类型`
- 情绪/节奏/结构不进这一条（它们进 L1 语境）

例：shot 1 的 breakdown = `[镜头景别:近景特写, 主体动作:筷子夹起顶层饼, 光影质感:暖光侧逆光, 构图方式:中心堆叠式构图]`
→ L2 query = `"近景特写 筷子夹起顶层饼 暖光侧逆光 中心堆叠式构图"`

**L3 — LLM 改写 query**（可选，`RAG_LLM_REWRITE=1` 开启）

调用轻量 LLM，把 `role + want + breakdown` 改写成一句"我需要什么样的用户素材"的自然表述，模仿 Split 的 `required_user_asset` 视角。

例：`"一段近景特写用户素材：手持食物成品展示，暖光突出油润质感，中心构图便于第一印象钩子"`

**默认策略**：L1 + L2（无网络额外调用）。L3 只在质量瓶颈期打开。

### 4.2 召回聚合：RRF（Reciprocal Rank Fusion）

**不采用 Split 的"每 query top-1"策略**，因为它会强行把不相关素材凑数。改用 RRF：

```python
def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: 每个 query 独立排序后的 asset_id 列表（高分在前）
    k: RRF 平滑常数，业界默认 60"""
    scores = {}
    for ranking in rankings:
        for rank, asset_id in enumerate(ranking):
            scores[asset_id] = scores.get(asset_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

流程：
1. 每条 query 独立 embed，与全库计算余弦，得到独立排序；
2. RRF 融合多个排序，得到最终 asset_id → score 映射；
3. 取 top-k（默认 5）作为 LLM 二判的候选。

RRF 的好处：
- 不需要归一化不同 query 的余弦分数（分布可能不同）；
- 出现在**多个 query 排名前列**的素材会自然浮上来，天然利于多维度匹配；
- 只在一条 query 中排名靠前的素材不会被完全淘汰，保留多样性；
- 参数少（只有 k=60，业界标准）。

### 4.3 阈值处理

- **去掉 embedding 侧的 0.62 / 0.45 硬阈值**；
- 保留一个最终兜底：如果 RRF 后 top-1 的 RRF 分数为 0（意味着所有 query 都没召回任何素材），直接返回 none；
- 判定权全部让给 LLM（skill `material_coarse_match`），LLM 给出 `status ∈ {direct, partial, none}`；
- LLM 失败时的兜底改为：有 top-1 就至少 partial，无候选才 none（不再依赖 embedding 分数）。

### 4.4 结果格式

```python
{
  "query_list": ["L1 query", "L2 query", ...],   # 便于日志与调试
  "matches": [
    {"id": "...", "rrf_score": 0.032, "cosine_scores": {"L1": 0.71, "L2": 0.65}, "meta": {...}},
    ...
  ],
  "count": 5,
  "store_size": 108,
  "backend": "qianfan"       # 或 "lexical_fallback"
}
```

`cosine_scores` 保留每条 query 的原始分数便于观测；`backend` 标识是否降级过。

---

## 5. 鲁棒性 / Fallback

**降级链**：

```
千帆 API embed
   ↓ RuntimeError (HTTP 5xx / timeout / rate limit)
本地 lexical fallback (BM25 或 char-level Jaccard)
   ↓ 依然为空
返回空 matches，由 LLM 兜底判 none
```

**Lexical fallback 实现思路**（简单够用即可）：
- 用 `jieba.cut_for_search` 对 query 与 key 都分词（keywords 也参与）；
- BM25 / TF-IDF 打分（`rank_bm25` 一行 pip 依赖）；
- 或最简版：字符级 Jaccard + keyword 命中数加权。

**开关**：`RAG_FALLBACK_TO_LEXICAL=1`（默认开）。

---

## 6. 配置项

新增可配置项：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `QIANFAN_API_KEY` | (必填) | 千帆凭证 |
| `EMBED_MODEL` | `qwen3-embedding-0.6b` | 模型名 |
| `EMBED_TIMEOUT` | `60` | 单次超时 |
| `EMBED_BATCH` | `32` | 单请求最大文本条数 |
| `EMBED_MAX_CHARS` | `500` | 单条截断长度 |
| `RAG_TOP_K` | `5` | LLM 二判候选数 |
| `RAG_LLM_REWRITE` | `0` | 是否开启 L3 LLM query 改写 |
| `RAG_FALLBACK_TO_LEXICAL` | `1` | 千帆失败是否降级到 lexical |
| `RAG_INCLUDE_SPEECH` | `0` | 是否把口播拼入 key |
| `RAG_INSTRUCTION` | `""` | Qwen3 instruction 模板（可选） |

---

## 7. 接口重构

**合并两条路径**，保留单一门面 `Retriever`：

```python
class Retriever:
    """统一的检索门面。取代 RetrievalTool + VectorStore 的重复实现。"""

    def __init__(self, task_id: str): ...

    # 入库
    def index_segments(self, segments: list, source: str = "") -> dict: ...

    # 检索（内部自动完成：多 query 生成 → embed(带 cache) → 各自排序 → RRF → 返回 top_k）
    def retrieve_for_shot(self, shot: dict, top_k: int = 5) -> dict: ...

    # 底层能力，供 PlanningAgent 的 FunctionTool 直接用
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def search(self, queries: list[str], top_k: int = 5) -> dict: ...

    # 管理
    def size(self) -> int: ...
    def clear(self): ...
```

保留旧类为薄兼容层，内部转发到 `Retriever`，逐步迁移调用方。

**Embedding 缓存**：进程内 `dict`，key = `f"{model}::{instruction}::{text}"`。避免同一 pipeline 里对同一 query 反复 embed（例如 3 个 shot 都要用同一个「行动号召·购买按钮特写」query）。

---

## 8. 可观测性

在关键点加 INFO 级日志（不需要落库）：

```python
_log.info("shot=%s queries=%s top1=%s score=%.3f backend=%s",
          shot_id, queries, top1_id, top1_score, backend)
```

配合已有的 `feasibility done: {direct, partial, none}` 汇总日志，出现 bad case 时能直接从日志回溯 query。

**可选**：PREFILTER_ONLY 调试模式（借鉴 Split），环境变量开启后跳过 LLM 二判，直接把 RRF 结果 dump 到 `outputs/debug/retrieval_dump_{task}.json`。

---

## 9. 与 Viral_Video_Split 的关键差异

| 维度 | Split | 本设计 |
|---|---|---|
| Backend | 本地 HF worker | 千帆 API + lexical fallback |
| 存储 | 内存 dict per-run | Per-task JSON（持久化，可复用） |
| Key 字段 | `one_sentence_summary` only | `summary + visual_description` |
| Query 生成 | `required_user_asset` 按标点拆 | L1(意图) + L2(视觉) + 可选 L3(LLM 改写) |
| 聚合 | 每 query top-1 去重 | **RRF 融合** |
| Fallback | Lexical 关键词 | Lexical BM25/Jaccard |
| LLM 判定 | ali-qwen3.7-plus 打分 (0-1) | material_coarse_match skill (status) |

**关键分歧点的理由**：
1. **Backend 不换千帆**：迁本地 HF 要 GPU/常驻子进程，Agent 是交互式服务，冷启+显存成本大；千帆 API 加 fallback 已能覆盖稳定性问题。
2. **存储持久化**：Agent 用户会反复调整参数重跑，per-task 缓存能显著省 embed 费用与延迟；per-task 隔离已足够避免污染。
3. **RRF 而非 top-1-per-query**：Split 的 top-1 策略在 query 之间语义高度重叠时会强凑不相关素材；RRF 更 robust，业界标准。
4. **Key 保留 visual_description**：Agent 上游 VLM 摘要质量参差，加一段视觉描述能兜住摘要过短/过泛的情况。

---

## 10. 迁移路线

**Phase 1（数据契约与隔离，低风险）**
- Key 字段砍到 `summary + visual_description`（`vectorstore.py._segment_text` / `retrieval.py._segment_text` 同步修改）
- Meta 扩展保留 VLM 全量字段（`asset_type / actions / visible_objects / visual_evidence_tags / keywords / quality_score / limitations`）
- Store 改 per-task 目录 `uploads/vectors/{task_id}/store.json`
- 从 orchestrator 复用 `rid` 作为 `task_id`，串到 `understand_materials` / `verify_materials`
- `VectorStore` 改 per-task 构造，`_VS` 模块级单例移除
- 删除旧 `uploads/vectors/store.json`（跨 task 污染，无迁移价值）
- `material_understanding.py` 显式生成 `seg["global_asset_id"]`，兑现 SKILL.md 的承诺
- `SKILL.md` 入库描述改写，与新 key 策略对齐
- 加基础 INFO 日志（`shot / queries / top1 / score`）

**Phase 2（召回策略升级，核心效果对齐 Split）**
- 多 query 生成（L1 意图 + L2 视觉，见 §4.1）
- RRF 融合聚合（见 §4.2）
- 去掉 embedding 侧硬阈值 0.62 / 0.45（`feasibility.py:28-29`），判定权全部让给 LLM 二判
- 补 embedding 空返回时的最终兜底（`_find_conflicts` / LLM 失败时）

**Phase 3（鲁棒性与工程化）**
- Lexical char-2gram Jaccard fallback（见 §12.3），千帆 API 失败自动降级
- Embedding 内存缓存（key = `f"{model}::{instruction}::{text}"`）
- 接口合并为单一 `Retriever`，`RetrievalTool` / `VectorStore` 退居兼容层

**Phase 4（可选增强，超越 Split 参考实现）**
- L3 LLM query 改写（Batch 模式，见 §12.4）
- PREFILTER_ONLY 调试模式（借鉴 Split）
- Qwen3 instruction 模板接入（`RAG_INSTRUCTION`）
- `asset_type` 软 pre-filter / `quality_score` tiebreaker 打分器落地

每个 Phase 都可独立上线验证，用 feasibility 输出的 `direct/partial/none` 分布做 A/B 观测。

---

## 11. 与用户素材理解 skill 的字段一致性

`user_material_understanding` skill（`skills_defs/user_material_understanding/SKILL.md`）的输出 schema 是本方案的上游数据契约，实施前必须先对齐。

### 11.1 字段覆盖情况

VLM 每个 `asset_segment` 实际产出以下字段（`SKILL.md:52-68`）：

| 字段 | required | 当前是否入库 | 本方案定位 |
|---|---|---|---|
| `asset_id` | ✅ | meta | 主键构成 |
| `source_video_id` | ✅ | meta | 主键构成 |
| `source_time_range` | ✅ | meta | 时间轴、`duration` 解析源 |
| `asset_type` | ✅ | ❌ 丢弃 | pre-filter / boost |
| `one_sentence_summary` | ✅ | key + meta | **key 主字段** |
| `visual_description` | ✅ | key + meta | **key 辅字段** |
| `speech_or_text` | 可空 | 当前误入 key | 移出 key（`RAG_INCLUDE_SPEECH` 可选打开） |
| `actions` | 可空 | ❌ 丢弃 | lexical fallback 词表 |
| `visible_objects` | 可空 | ❌ 丢弃 | lexical fallback 词表 |
| `visual_evidence_tags` | 可空 | ❌ 丢弃 | LLM 二判提示 |
| `keywords` | ✅ (4-8 个) | 当前误入 key | 移出 key，进 meta 供 lexical fallback |
| `quality_score` | ✅ | ❌ 丢弃 | RRF 后 tiebreaker |
| `limitations` | 可空 | ❌ 丢弃 | 观测/调试 |

**结论**：字段完全够用，本方案要求的 `summary + visual_description` 两个 key 字段都是 skill 层面 required。此外还有 6 个字段（`asset_type / actions / visible_objects / visual_evidence_tags / quality_score / limitations`）当前被 `vectorstore.py:97-103` 的入库逻辑丢弃，本方案需要保留到 meta。

### 11.2 需要修正的两个不一致

#### 11.2.1 SKILL.md 的入库描述与新 key 策略冲突

`skills_defs/user_material_understanding/SKILL.md:25` 目前写：

> 入库要求：每完成一个素材的理解，产出的每个片段都必须按字段（**one_sentence_summary / visual_description / keywords / speech_or_text** 等）写入向量库

—— 显式点名 `keywords / speech_or_text` 也入库，与本方案 §3.2「key 只用 summary + visual_description」冲突。

**修正**：改为

> 入库要求：每完成一个素材的理解，产出的每个片段都必须调用 index_segments 写入向量库；语义索引默认由 `one_sentence_summary + visual_description` 生成，其余字段（keywords / speech_or_text / asset_type / actions / visible_objects / visual_evidence_tags / quality_score 等）作为 meta 一并保存，供后续 lexical fallback 与 LLM 二判使用。

#### 11.2.2 `global_asset_id` 承诺未兑现

`SKILL.md:78` 承诺：

> 后处理会为每个片段补 global_asset_id（形如 source_video_id::asset_id），模型无需输出。

但 `material_understanding.py:155-158` 只做了 `seg.setdefault("source_video_id", label)`，**没有真正生成 `global_asset_id`**。当前是靠 `vectorstore.py:89` 的兜底 `gid = seg.get("global_asset_id") or f"{source}::{asset_id}"` 硬凑。

**风险**：
- 同一素材若通过不同 `source`（相对路径 vs 文件名）入库，`gid` 会不一致，去重失效；
- 下游 `feasibility.matched_asset_id`（如 `"382fd4989715cd36_牛肉饼4-素材4.MP4::A1"`）依赖此 id 回查，一旦 source 漂移就查不回来。

**修正**：在 `_understand_one`（`material_understanding.py:155-158`）里显式生成 `seg["global_asset_id"] = f"{seg['source_video_id']}::{seg['asset_id']}"`，把 SKILL.md 说的"后处理"真正落到代码里，`vectorstore.py` 的兜底逻辑保留但退居次位。

### 11.3 额外的利用建议

以下 VLM 已产出但当前未利用的字段，建议在本方案落地时一并接入：

- **`asset_type`（商品特写/使用演示/人物口播/空镜）**：在 RRF 前做**软 pre-filter**——例如 shot.role=`开场钩子` 时给 `asset_type ∈ {商品特写, 使用演示}` 的候选加权 1.2×，不完全排除其他类型。
- **`visual_evidence_tags`**：`compact_asset_for_match`（若参考 Split 的做法）打包给 LLM 二判时必须带上，等价于 Split 里的同名字段，是 LLM 判 direct/partial/none 的关键证据。
- **`quality_score`**：RRF 融合后同分 tiebreaker，或做轻微加权 `final_score = rrf_score * (0.7 + 0.3 * quality_score)`。
- **`actions + visible_objects + keywords`**：合并成 lexical fallback 的 BM25/Jaccard 词典，比重新 `jieba.cut` 更可控。
- **`limitations`**：不参与召回，但落到 log 与 debug dump，便于分析 bad case。

---

## 12. 开放问题的确定方案

### 12.1 task_id 从哪里传入

**方案**：复用 orchestrator 已有的 `rid = obs.new_request_id()`（`src/agent/orchestrator.py:110/345`），显式串到下游，不引入新概念。

- 改函数签名：
  ```python
  async def understand_materials(materials, *, use_cache=True, concurrency=None, task_id: str = "")
  async def verify_materials(shots, *, max_agents=None, task_id: str = "")
  ```
- `VectorStore` 从模块级单例改为 per-task 构造，`_VS = VectorStore()` 从模块级删除，两个 agent 函数入口按 `task_id` 实例化：
  ```python
  class VectorStore:
      def __init__(self, task_id: str = ""):
          self._store_path = os.path.join(STORE_DIR, task_id or "_shared", "store.json")
  ```
- Cache 隔离策略：**只让 vector store 分 task，`cache.set/get("material", ...)` 保持全局共享**（同一素材 URI 的 VLM 理解结果本身与 task 无关，跨 task 复用能省 VLM 调用费）。

### 12.2 旧 store.json 直接删除，不迁移

- 现有 100+ 条历史是"跨 task 混合污染"的产物，无正确归属；
- Feasibility 检索的是"本任务 shot vs 本任务用户素材"，历史素材本就属于其他上传者，回查不到不影响任何下游；
- Phase 1 上线脚本：`rm -rf uploads/vectors/store.json`，并把 `uploads/vectors/` 加进 `.gitignore`（如未添加）。

### 12.3 Lexical fallback 用极简自实现，不引入新依赖

- 现有 `requirements.txt` 只 2 行（`agentscope` + `python-frontmatter`），引入 `jieba` + `rank_bm25` 意味着 20MB+ 依赖与首次加载词典延迟；
- Fallback 触发频率低（只在千帆 API 挂时），性价比不高；
- 中文场景 2-gram 字符级 Jaccard 无需分词即可工作：
  ```python
  def _char_ngrams(text: str, n: int = 2) -> set[str]:
      s = text.replace(" ", "")
      return {s[i:i+n] for i in range(max(0, len(s) - n + 1))}

  def lexical_score(query: str, key_text: str, keywords: list[str]) -> float:
      q, k = _char_ngrams(query), _char_ngrams(key_text)
      jaccard = len(q & k) / max(1, len(q | k))
      hits = sum(1 for kw in keywords if kw and kw in query)
      return jaccard + 0.1 * hits   # keywords 显式命中加 boost
  ```
- 未来若确需 BM25，作为可选依赖 `try import rank_bm25` 降级导入，不强制。

### 12.4 L3 LLM query 改写延后到 Phase 4，不新增 skill

- Phase 1-3 全部不做 L3，先验证 RRF + L1/L2 效果；
- 每 shot 一次额外 LLM 调用会显著拉高交互式 pipeline 延迟；
- Phase 4 落地时用 **Batch 改写**：一次性把整个 replication plan 的所有 shot 送给 LLM，均摊延迟，复用现有 `LLM_MODEL`，不引入新模型/新推理路径。
