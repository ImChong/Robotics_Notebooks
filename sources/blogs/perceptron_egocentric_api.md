# perceptron_egocentric_api

> 来源归档（blog）

- **标题：** Introducing Perceptron Egocentric API
- **类型：** blog
- **来源：** Perceptron Inc. 官方博客
- **原始链接：** https://www.perceptron.inc/blog/introducing-perceptron-egocentric-api
- **产品页（Notion）：** https://app.notion.com/p/perceptron/398528411b178076a9fed23b9977f065
- **入库日期：** 2026-07-10
- **最后更新：** 2026-07-10
- **一句话说明：** Perceptron Egocentric 是基于 Mk1 具身感知模型的机器人/第一人称视频自动标注 API，在 WGO-Bench 上超越 Gemini Robotics ER-1.6 与 Gemini 3.5 Flash 驱动的 Macrodata 管线，输出原子操作分段、子任务语义标签与双手稠密 grounding。

## 核心摘录

### 1) 产品定位与输出形态

- **输入：** 原始机器人演示或 egocentric 视频（可选 episode 级任务指令）。
- **输出：** 可喂策略训练的 **结构化监督**：
  - 时间轴上的 **原子操作事件（atomic manipulation events）** 分段；
  - **自洽子任务（self-contained subtask）** 文本标签；
  - **双手稠密 grounding**（检测框 + 21 关键点骨架 + 左右手独立动作分类）。
- **状态（2026-07）：** Early Access，面向机器人实验室、研究者与数据供应商的大规模标注管线。

### 2) 三大能力模块

1. **Dense hand-annotation（Mk1 sidecar）**
   - 逐帧双手 bbox + **21 关键点**（腕 + 每指 4 关节），带左右身份；
   - 操作边界跟随 **手/接触** 而非像素变化；
   - 预定义 manipulation taxonomy：reaching、grasping/pinching、lifting、holding、placing/inserting、pushing/pulling、rotating、opening/closing、releasing 等 + visibility 状态。

2. **Granular sub-task annotation**
   - 将机器人或人类演示拆为原子动作，生成 caption 并校验，强调 **时间边界 + 语义自洽**。

3. **Instruction-optional**
   - **With instruction：** 以 episode 任务指令为上下文；
   - **No instruction（blind）：** 无任务先验，从视频恢复子任务结构。

### 3) WGO-Bench 评测（Macrodata「What's Going On」）

- **数据：** HomER（egocentric）、DROID（外置机器人相机）、Galaxea（机器人头相机）；**743** 人工 gold 段、**62** 条任务指令、**71.5** 分钟视频。
- **匹配协议：** temporal IoU ≥ 0.75；标签正确性由 benchmark 规定的 **LLM judge**（与 Macrodata 公开评分一致）；**semantic end-to-end F1** = 边界 + 标签同时正确。
- **对照：** Macrodata WGO 管线（Gemini 3.5 Flash + Gemini Robotics ER-1.6）；one-pass（分段与标注同 pass）与 seeded-relabeling（每段二次 Gemini 标注）两变体。

### 4) 核心结果（博客摘要，2026-07）

| 指标 | Perceptron Egocentric（with instruction） | WGO one-pass | 相对变化 |
|------|-------------------------------------------|--------------|----------|
| Semantic end-to-end F1 | **0.280** | 0.158 | **+77%** |
| Segment F1 | **0.370** | 0.302 | +23% |
| Semantic precision | **0.330** | 0.190 | — |
| Semantic recall | **0.244** | 0.136 | ~+80% 相对 |
| 完全正确 gold 段数 | **181 / 743** | ~101 | — |

- **成本：** with-instruction 约为人工标注（~**$50/视频小时**）的 **1/10–1/15**；仍低于 Macrodata 全管线成本。
- **No instruction：** 最佳精度配置在分割与端到端上仍超先前 SOTA；**fast-inferred** 配置降本 ~30%，end-to-end F1 **0.182**，仍高于 WGO no-instruction seeded（**0.138**）。
- **WGO seeded-relabeling：** 条件标签准确率 78.1%，但 end-to-end F1 仅 **0.168**，低于 Perceptron with-instruction 无 relabeling 的 **0.280**。

### 5) 为何有效（作者论点）

- 底座 **Mk1** 在预训练规模上原生建模 **手、接触、物体状态、空间关系**；**未**在 WGO 评测 harness、taxonomy 或 episode 上微调。
- 管线为 **推理期框架**，能力来自 **具身推理** 而非任务专用训练；通用 VLM 管线「抽帧猜发生了什么」，Mk1「感知正在发生的操作」。

### 6) 成本说明（博客脚注）

- 不含 WGO-Bench **judge 评分成本**（评测成本，非标注成本）；
- 手部姿态流在 Perceptron 自有 GPU 运行，启用约 **+$1/视频小时**（非 metered API）；
- 成本自 2026-07 标准 list price 用量估算，不含 batch 折扣。

## 对 wiki 的映射

- [perceptron-egocentric](../../wiki/entities/perceptron-egocentric.md) — 产品/实体页（新建）
- [auto-labeling-pipelines](../../wiki/methods/auto-labeling-pipelines.md) — 代表性 VLM/具身自动标注管线
- [gemini-robotics](../../wiki/entities/gemini-robotics.md) — WGO 对照基座之一
- [imitation-learning](../../wiki/methods/imitation-learning.md) — 子任务分段监督 → BC/VLA 数据准备
- [vla](../../wiki/methods/vla.md) — 语言–视觉–动作数据引擎语境

## 当前提炼状态

- [x] 博客核心摘要与 WGO 数字摘录
- [x] wiki 实体页映射确认
- [ ] 若 Macrodata/WGO 公开技术报告入库，可补独立 benchmark 页或对比页
