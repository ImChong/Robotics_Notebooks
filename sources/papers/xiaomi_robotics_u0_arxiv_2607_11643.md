# Xiaomi-Robotics-U0（统一具身合成世界基础模型）

> 来源归档（ingest）

- **标题：** Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model
- **类型：** paper（arXiv）
- **原始链接：**
  - <https://arxiv.org/abs/2607.11643>
  - <https://arxiv.org/html/2607.11643v1>
- **项目页：** <https://robotics.xiaomi.com/xiaomi-robotics-u0.html>
- **代码：** <https://github.com/XiaomiRobotics/Xiaomi-Robotics-U0>
- **机构：** Xiaomi Robotics
- **发布日期：** 2026-07（arXiv v1）
- **入库日期：** 2026-07-15
- **一句话说明：** **38B** 自回归 **世界基础模型** 将 **foundation T2I/X2I** 与 **多视角具身场景 / 迁移 / 视频** 置于同一 NTP 目标；结构化五维场景控制 + **FlashAR+** 工程加速；**WorldArena #1** 与 **π₀.₅** 真机 OOD **+26.3 pts** 验证「WM 即数据引擎」。

## 核心摘录（MVP）

### 1) 问题：后训机器人数据会削弱 foundation 泛化

- **摘录要点：** 现有具身 WM 多在 **纯机器人轨迹** 上持续微调，牺牲预训练阶段的 **多样 T2I/X2I** 任务，导致生成多样性、可控性与可扩展性受限。U0 把 **具身合成** 表述为 **foundation 图像/视频生成的自然延伸**，在 **通用域 + 具身域** 上 **统一 AR 目标** 共训。
- **对 wiki 的映射：**
  - [Xiaomi-Robotics-U0](../../wiki/entities/xiaomi-robotics-u0.md) — 动机与统一 formulation。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 「保留 foundation 能力」的持续训练范式。

### 2) 能力矩阵：单步 + 序列

- **摘录要点：** **单步：** T2I、X2I、多视角 **Embodied Scene Generation**、**Embodied Transfer**（深度条件 + 编辑描述 → 多视角 RGB）。**序列：** 子任务–子目标图文交错；**1/3/5 FPS** 操纵视频；场景合成可 **rollout** 为时序视频，形成 **agentic 数据引擎**。
- **对 wiki 的映射：**
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 静态合成 → 轨迹生成闭环。
  - [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) — 可控视频 / 数据增广层。

### 3) 架构与 FlashAR+

- **摘录要点：** 基于 **EMU3.5**（**Qwen3-32B** decoder + **IBQ** 16×16 图像 token）；全任务 **无专用预测头**。**FlashAR+** 在目标图像区 **反对角并行** 解码，前缀（文本 + 参考图）保持 AR；**vLLM** 批调度 + paged KV；1024² 单样本 **450.77s → 5.44s**（FlashAR+ + vLLM）。
- **对 wiki 的映射：**
  - [Xiaomi-Robotics-U0](../../wiki/entities/xiaomi-robotics-u0.md) — 架构表与推理加速节。

### 4) 数据与五维结构化控制

- **摘录要点：** **9.5M** 单步 / **2.6M** 序列 clip；**Qwen3-VL-235B** 标注：通用场景字幕、**五维具身场景**（workspace / task objects / irrelevant / lighting / background）、**Video Depth Anything** 几何、**HDBSCAN** 轨迹子任务切分。Embodied transfer 用 **多视角深度 + 场景描述** 监督 **多视角 RGB**。
- **对 wiki 的映射：**
  - [Open X-Embodiment](../../wiki/concepts/open-x-embodiment.md) — 异构数据来源之一（AgiBotWorld、OXE、MiBot 等）。

### 5) 评测与真机增广

- **摘录要点：** 多视角场景 / 迁移：**人类 pairwise** 优于 **GPT-Image-2**（Easy/Hard win rate 见项目页）。**WorldArena EWMScore_P 73.64 #1**（UNIS 条目）。真机三任务：用 **结构化 transfer** 对示教做背景/光照/工作区增广，**π₀.₅** 在干扰组完成度 **36.9%→63.2%**，基线到干扰降幅 **44.1→18.9 pts**。
- **对 wiki 的映射：**
  - [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md) — 同生态 VLA；U0 侧重复数据引擎叙事。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 桌面灵巧 / 可变形物体任务语境。

## 当前提炼状态

- [x] arXiv abs + HTML + 官网 + GitHub 已对齐摘录
- [x] wiki 映射：`wiki/entities/xiaomi-robotics-u0.md` 新建，并与 generative-world-models / xiaomi-robotics-0 交叉引用
