# Gemini Robotics 2 brings whole body intelligence to robots

> 来源归档（blog / Google DeepMind 官方）

- **标题：** Gemini Robotics 2 brings whole body intelligence to robots
- **类型：** blog
- **作者 / 组织：** Carolina Parada / Google DeepMind（Gemini Robotics team）
- **原始链接：** <https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/>
- **发表日期：** 2026-07-30
- **入库日期：** 2026-07-31
- **抓取方式：** 官方页直连（WebFetch + HTML 交叉核对）
- **一句话说明：** Google DeepMind 发布 **Gemini Robotics 2** 三件套——全身人形 **VLA**、具身推理 **ER 2**、端侧 **On-Device 2**——把能力叙事从桌面操作推进到 **feet-to-fingertips 全身控制、多末端灵巧、多机协作与数小时级跨本体适配**；ER 2 公开预览，VLA/On-Device 仍 early-access。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-07-31） |
|----|-------------------------|
| 产品页 | <https://deepmind.google/models/gemini-robotics/> → 归档 [sources/sites/gemini-robotics.md](../sites/gemini-robotics.md) |
| ER 2 开发者博文 | <https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/> |
| ER 2 Model Card | <https://deepmind.google/models/model-cards/gemini-robotics-er-2/> |
| 安全技术报告 | <https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-2-Safety.pdf> |
| ER 2 API / AI Studio | Public preview；模型串示例 `gemini-robotics-er-2-preview`（见 [ai.dev](https://ai.dev/prompts/new_chat?model=gemini-robotics-er-2-preview)）；文档 [Robotics overview](https://ai.google.dev/gemini-api/docs/robotics-overview)、[Live API](https://ai.google.dev/gemini-api/docs/live-api) |
| 编排样例代码 | **已开源（部分）**：[`google-gemini/robotics-samples`](https://github.com/google-gemini/robotics-samples)（Apache-2.0）→ [sources/repos/google-gemini-robotics-samples.md](../repos/google-gemini-robotics-samples.md) |
| VLA / On-Device 权重与训练 | **未公开发布**（early-access partners / Trusted Tester）；不可本地复现全身 VLA |
| ASIMOV-Agentic | 博客宣称新 agentic 安全基准；HF 入口见 `google/asimov_agentic`（访问可能需登录/授权，以页面为准） |
| 可信度边界 | 官方产品博客 + 安全 PDF；成功率多为自报演示条形图，非独立 peer-reviewed 榜 |

## 核心摘录（归纳，非全文）

### 三模型分工

| 模型 | 角色 | 访问（博客声明） |
|------|------|------------------|
| **Gemini Robotics 2** | 最强 **VLA**：视觉+语言 → 电机控制；可控全身人形与双臂；手/夹爪灵巧 | early-access partners |
| **Gemini Robotics ER 2** | 最强 **ER / VLM agent**：人机对话、物理世界理解、数分钟级多步规划、多机协作 | Google AI Studio + Gemini API public preview；Enterprise Agent Platform private preview |
| **Gemini Robotics On-Device 2** | 端侧高效 VLA；继承 1.5「motion transfer」；新双臂本体数小时适配（常 <200 例） | early-access / Trusted Tester |

### 能力轴（相对前代）

1. **全身控制（whole-body）**：首次强调同一 VLA 控制整台人形（feet → fingertips）；示例 Apptronik Apollo 2「把浇水壶放到底层绿色垃圾桶」——行走、取放、货架放置；博客承认运动速度仍待提升。
2. **灵巧操纵**：Apollo 2 + SharpaWave 五指手（22 DoF）打结、封 ziplock；Franka Duo + Robotiq 夹爪紧凑装箱；**多指灵巧仍最难**（博客图示 medium–high 于全身/夹爪，多指任务方差大）。
3. **Agentic ER + 多机**：ER 2 作高层脑，协调 VLA、跟踪进度、失败自纠；任务可持续数分钟、数百决策；**多机协作**（异构机器人分工）；开发者博文补充 progress classification（自报 57.4%）与 moment-finding（自报 91.3%、MAE ~0.96s）。
4. **On-device 跨本体**：同一叙事下适配 Dexmate / SO101 / Trossen 等形态差异大的双臂平台。
5. **安全**：多层物理+AI 安全；引入 **ASIMOV-Agentic**（拒不安全 tool call、可行性预判、不确定时请求人类）；ER 2 在 safety constraint / human proximity 上自报为迄今最安全。

### 同一 checkpoint 跨本体演示

博客强调 **同一 VLA checkpoint** 驱动：

- Apptronik Apollo 2 + SharpaWave 手
- Apollo 2 + Inspire 手
- Franka Duo + Robotiq 夹爪

并按 skill category 给出平均成功率条形图（具体数值以页面图为准，勿二次转述为精确 SOTA）。

### 合作伙伴

Apptronik、Boston Dynamics、Agile Robots（致谢段）；ER 2 Live API 样例含 Spot 导航/操作编排。

## 对 wiki 的映射

- [gemini-robotics](../../wiki/entities/gemini-robotics.md) — 本篇主升格实体页（GR2 迭代）
- [vla](../../wiki/methods/vla.md) — 闭源全身 VLA 产品对照
- [foundation-policy](../../wiki/concepts/foundation-policy.md) — 通才策略闭源对照
- [whole-body-control](../../wiki/concepts/whole-body-control.md) — 学习式全身控制 vs 经典 QP WBC
- [loco-manipulation](../../wiki/tasks/loco-manipulation.md) — 人形走取放任务族
- [hub-cross-embodiment](../../wiki/overview/hub-cross-embodiment.md) — On-Device 数小时跨本体适配

## 可信度与使用边界

- **官方营销 + 技术叙事**，定量多为内部评测/演示；引用时标注读取日期与「自报」。
- **不要**把 VLA/On-Device 写成可本地训练基线；可复现入口目前主要是 **ER 2 API + robotics-samples 编排**。
- 「全身智能」≠ 已替代经典 WBC/低层跟踪栈；博客亦承认速度与多指精度仍有差距。

## Citation

```bibtex
@misc{deepmind2026geminirobotics2,
  title        = {Gemini Robotics 2 brings whole body intelligence to robots},
  author       = {Parada, Carolina and {Gemini Robotics Team}},
  howpublished = {Google DeepMind Blog},
  year         = {2026},
  month        = {7},
  url          = {https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/},
  note         = {Accessed 2026-07-31}
}
```
