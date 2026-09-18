# arXivisual 工具站（一手入口归档）

> 来源归档（ingest）

- **标题：** arXivisual — arXiv Papers, Visualized
- **类型：** site / tool
- **原始链接：** <https://arxivisual.org>
- **代码：** <https://github.com/rajshah6/arXivisual>
- **入库日期：** 2026-09-18
- **一句话说明：** 免费 Web 工具：把 arXiv 预印本变成带 AI Manim 短片与语音的 scrollytelling 阅读体验；适合快速扫论文结构与核心概念（非 peer review 替代）。

## 核心摘录（策展）

### 1) 定位：论文 monolith → 可观看的 visual story

- **摘录要点：** 研究论文常以 dense PDF/HTML 呈现；arXivisual 用 **分段阅读 + 嵌入动画** 降低入门门槛，风格接近 3Blue1Brown 式 Manim 讲解。
- **对 wiki 的映射：**
  - [arXivisual](../../wiki/entities/arxivisual.md) — 工具实体。
  - [Manim](../../wiki/entities/manim.md) — 底层动画引擎。

### 2) URL 零摩擦入口

- **摘录要点：** `arxiv.org/abs/1706.03762` → `arxivisual.org/abs/1706.03762`（在 `arxiv` 后加 `isual`）；首页亦支持粘贴 ID/URL。
- **对 wiki 的映射：**
  - [arXiv](../../wiki/entities/arxiv.md) — 上游预印本层。

### 3) 多智能体管线与质量门

- **摘录要点：** Ingest（arXiv/ar5iv）→ Analyze → Plan → Generate Manim → **四道 validation**（syntax / spatial / voiceover / render，最多 5 轮重试）→ Render + TTS → 存储。Langfuse 可观测 token 与成本。
- **对 wiki 的映射：**
  - [arXivisual](../../wiki/entities/arxivisual.md) — 流程图与工程实践。

### 4) 开源与部署边界

- **摘录要点：** GitHub 可本地起前后端；生产托管在 **Azure Container Apps** + **Cloudflare R2**。**截至 2026-09-18 仓库无 LICENSE 文件**——商用/二次分发前需自行确认。
- **对 wiki 的映射：**
  - [arxivisual 仓库](../repos/arxivisual.md)
  - [arxivisual 项目页](../sites/arxivisual-org.md)

## 当前提炼状态

- [x] 站点 / GitHub README / 开源核查已完成
- [x] wiki 映射：`wiki/entities/arxivisual.md`
