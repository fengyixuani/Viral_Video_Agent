# 长 AI 带货短剧复刻功能 —— 设计与开发日志

> 目标：输入一个爆款 AI 带货短剧（先支持 ~1 分钟），复刻出**剧情套路一致、但表面不雷同（不被一眼看出抄袭）**的新短剧，并把商品替换为指定商品（本次：理然 MAKE SENSE 去黑头泥膜棒）。
> 场景：在「选择创意短剧场景」时使用。
> 架构：遵循本仓库既有 Agent 规范（AgentScope 2.0.4 + wenchain 网关 + SSE 流式 + `src/tools` 业务层 + `skills_defs` 技能定义）。

## 一、链路（Pipeline）

```
参考短剧视频
  │  ① 理解 understand      (qwen3.7-plus 视觉)   → 核心成分 JSON（套路/人物/场景/分镜/节奏/卖点）
  ▼
核心成分
  │  ② 写脚本 script        (qwen3.7-max 文本)    → 新剧本 + 人物设定 + 分段(≤15s)分镜脚本
  ▼
剧本 + 人物设定
  │  ③a 角色三视图 sheets   (seedream T2I)        → 每个人物 front/side/back 三视图（公网 url，人物一致性锚点）
  │  ③b 分镜图 storyboard   (seedream i2i)        → 每段 1 张关键帧图，参考=角色三视图(+商品图)
  ▼
分镜图 + 分段 prompt
  │  ④ 生成视频 video       (seedance 2.0 i2v)    → 每段 ≤15s 片段（首帧=分镜图）
  │  ④' 拼接 concat         (ffmpeg)              → 最终整片
  ▼
最终视频
  │  ⑤ 验证 verify          (qwen3.7-plus 视觉)   → 对照目标打分；未达标回到②/③/④迭代
  ▼
达标成片
```

## 二、关键能力验证（wenchain 网关，已实测 ✅）

网关：`http://wenku-openai.baidu-int.com/wenchain/strategy/incommonuserr`
鉴权：`channel = wangpantob_all_video_copy`（无独立 token，channel 即凭证）

- ✅ 文生图 T2I：`doubao-seedream-5-0-260128`，`seedream_options{prompt,size,n,response_format:url,watermark:false}`，返回公网 `bos_url`。有效 size 需满足最小像素，实测 `2048x2048` OK。
- ✅ 图生图 i2i：同模型，`seedream_options.image = [data_url,...]`。**实测支持 base64 data URL 作为参考图，且支持多张参考图**（角色三视图 + 商品图可同时喂）。输出仍是公网 `bos_url`。→ 解决了「本地商品图无公网 URL」的问题，无需额外 BOS 上传器。
- ✅ 文生视频 T2V：`doubao-seedance-2-0`，`seedancepro_options.content=[{type:text,text:"<prompt> --ratio 9:16 --dur N"}]`，返回公网 `video_url`。时长向上取整并夹到 [4,15]s。
- 图生视频 i2v：seedance content 里加 `{type:image_url,image_url:{url:<公网url>},role:first_frame}`。首帧用 seedream 输出的公网 bos_url（base64 首帧会超时，故必须公网）。
- ✅ 视觉理解/验证：`as_core.complete_json(system,user,vision=True,media=[{type:video/image,url:path}])` → qwen3.7-plus。大视频自动降分辨率转码（≤18MB base64）。
- ✅ ffmpeg：经 `imageio_ffmpeg` 提供（无系统 ffmpeg），用于拼接与转码。

## 三、人物一致性策略

seedance 单段仅 15s，长剧需多段拼接，跨段人物必须一致：
1. 先由剧本为**每个人物**生成**角色三视图**（front/side/back，同一 prompt 锚定外观/发型/服装/配色）。
2. 每段**分镜图**用 seedream **i2i**，参考图 = 该段出场人物的三视图（+ 商品段附带商品图），保证同一人物在不同分镜里外观一致。
3. 每段视频用 seedance **i2v**，首帧 = 该段分镜图，保证段内外观延续分镜图。
4. 分镜图 prompt 显式复述人物固定特征（发型/脸型/服装/配色），进一步约束。

## 四、防「一眼抄袭」策略

复刻的是**套路/结构**（story beats、hook 形式、卖点顺序、节奏、转化闭环），不是逐帧临摹：
- 保留：叙事结构、情绪曲线、卖点编排、CTA 逻辑、镜头节奏密度。
- 改写：具体台词、人物形象、场景布置、镜头构图、配色、道具，商品替换为目标商品。
- 写脚本阶段显式要求「保留套路、重构表层、避免与原片镜头一一对应」。

## 五、代码结构

- `src/tools/wenchain_media.py`：底层媒体客户端（T2I / i2i / T2V / i2v / 下载 / ffmpeg 拼接）。纯业务层，无 Agent 逻辑。
- `src/drama/`：长剧复刻业务
  - `understand.py` ①理解
  - `script.py` ②脚本+人物
  - `storyboard.py` ③三视图+分镜图
  - `video.py` ④生成+拼接
  - `verify.py` ⑤验证
  - `pipeline.py` 编排（SSE 流式，串起 ①~⑤ 与迭代）
- `skills_defs/drama_replication/SKILL.md`：技能定义（创意短剧场景）。
- server 侧新增 `/api/drama_replicate` SSE 入口。

## 六、开发进度日志

- [进行中] 能力验证完成，开始搭建 `wenchain_media` 客户端与理解阶段。
- 理解①实测：参考片实为 28s「孙悟空 vs 如来 护手霜反转剧」，qwen3.7-plus 成功拆出套路
  「经典IP冲突+意外反转+自然植入」、2 个人物、5 个镜头。
- 脚本②实测：保留套路、重构为「哪吒闹海变护肤现场」，商品替换为泥膜棒，differentiation 清晰。
- 三视图③a + 分镜图③b 实测：2 张角色三视图 + 5 张分镜图（i2i，含商品段用商品图作参考）全部成功。
- **坑：seedance i2v 对写实真人首帧触发风控（40000002）**。因参考片本就是 AI 漫剧，
  已把全链路统一为 **3D 动画/国漫风格（明确非真人）**，实测 i2v 通过。
- **健壮性**：i2v 单段失败时 → 重试 → 重画更卡通首帧再试 → 回退 t2v（纯文本），
  避免单段失败拖垮整条流水线（`src/drama/video.py`）。
- Agent 规范接入：新增 `skills_defs/drama_replication/SKILL.md`（创意短剧场景，已被
  `skills.skill_list()` 识别）与 server `/api/drama_replicate` SSE 入口。
- **新增 ASR 能力**（`src/drama/asr_util.py`）：
  - 理解①：先对参考视频做 ASR（本机离线 Qwen3-ASR-0.6B，走 `tools.asr.ASRTool`+`asr_cache`
    共享缓存；不可用时回退 qwen3.7-plus 听写），把口播文本喂进理解 prompt 并写入 `core.asr`，
    让台词/卖点话术还原更准。
  - 验证⑤：对成片做 ASR，把口播文本 `final_video_asr` 一并交给 qwen3.7-plus，
    新增打分维度 `audio_script_match`（口播是否契合剧本台词/卖点并完成转化引导）。
- **迭代收敛**：`run.py --iters` 默认 4（每轮 verify `pass` 即提前结束），实现「未达标就一直改」。

## 七、v2 需求变更（分镜=故事板 / 更细分镜 / 台词发声 / ASR 重理解）

用户反馈三点，处理如下：
1. **分镜脚本 = 故事板 + 更细分镜**（已改代码）：
   - 脚本②改为「段(segment) → 段内 2~4 个分镜格(shot)」，每个 shot 4~6s、含独立台词。
   - 新增 `gen_story_boards`：每段生成一张多格漫画式故事板整图（格子+描述），供评审。
   - 每个分镜格单独出关键帧 → seedance i2v 首帧；视频逐分镜格生成后拼接。
2. **理解加 ASR + 重新理解**（已改代码）：understand① 先 ASR 再理解，台词对齐 ASR。
   - ⚠️ 本机 Qwen3-ASR 环境损坏（`transformers` 版本不兼容：`cannot import name 'GenerationMixin'`），
     已自动回退 qwen3.7-plus 听写；对有清晰人声的参考片转写正确（"如来/你的手有点滑啊/小金人牌…"）。
3. **成片要有人说话**（✅ 已确认可用）：
   - seedance「把台词写进 prompt」**确实会生成人声对白**（用户实听确认；ffmpeg 客观测音
     mean≈-18dB 非静音）。之前误判为"无人声"是因为**qwen 是视觉模型、听不了音频**，不能用于音频验证。
   - 因此人声走 seedance prompt 内嵌台词即可（`video._shot_prompt` 已内嵌台词并要求发声），无需额外 TTS。

## 八、ASR 修复（用对 conda 环境 + 清理子进程环境污染）

- 正确的项目 conda 在 `/root/chengzhiyang/miniconda3`（见 SETUP.md），ASR 专用环境
  `qwen3-asr-cu128`（torch 2.8+cu128，CUDA 可用；`viral-asr` 的 cu130 与驱动不匹配、CUDA 不可用）。
- 直接用该解释器跑 vendored `run_qwen3_asr_test.py` 本就正常。之前在流水线里失败的真因是
  **子进程继承了调用方的 `PYTHONPATH=src` 等变量，污染了 ASR 解释器的导入**，报
  `cannot import name 'GenerationMixin'`。
- 修复：`tools/asr.py`（及 `agent_edit/tts.py`）在起子进程前 `pop` 掉
  `PYTHONPATH/PYTHONHOME/PYTHONSTARTUP`。修复后流水线内 ASR 正常：
  转出参考片真实台词「如来，我已去天边尽头做了标记…是小金人牌高奢丝滑护手霜，尽享手部丝滑。」（4 段带时间戳）。
- 理解①与验证⑤现在都用**真实音频 ASR**（不再依赖 qwen 读字幕）。




