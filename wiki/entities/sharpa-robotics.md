---
type: entity
tags: [company, dexterous-hand, tactile-sensing, data-flywheel, humanoid, sharpa-robotics]
title: 夏帕机器人（Sharpa Robotics）
status: complete
updated: 2026-10-16
related:
  - ./paper-sata.md
  - ./paper-tacmap.md
  - ./paper-dexemg.md
  - ./paper-ftp-1.md
  - ./paper-do-as-i-do.md
  - ./paper-play2perfect.md
  - ./paper-dex-x.md
  - ./paper-tacbpm.md
  - ./paper-uvta-human-tactile-dexterous-manipulation.md
  - ./paper-galatea.md
  - ./simtoolreal.md
  - ./paper-trex-tactile-reactive-dexterous-manipulation.md
  - ./paper-sa-2603-08122-mode-vla-mixture-of-dexterous-experts-vla.md
  - ./paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div.md
  - ./paper-wm-craftnet.md
  - ./paper-artmanip.md
  - ./paper-teledexter.md
  - ./paper-spd.md
  - ./paper-poise-in-hand-6d-pose-reaching.md
  - ./sharpadex.md
  - ./craftnet.md
  - ./sharpa-d01-robot.md
  - ./sharpa-w02-hand.md
  - ./sharpa-ae01-data-glove.md
  - ../overview/sharpa-academic-route-three-lines.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/visuo-tactile-fusion.md
sources:
  - ../../sources/blogs/wechat_sharpa_dexterous_manipulation_2026-10-16.md
  - ../../sources/blogs/wechat_tiannan_sharpa_academic_route_2026-10-22.md
summary: "夏帕机器人（Sharpa Robotics）：2024 年成立、总部新加坡的灵巧操作公司，以 22 自由度触觉灵巧手 SharpaWave 为平台；学术线主张「数据→接触→跨本体表征」三条主线，用 System 0/1/2 分层把接触交给高频触觉底层。"
---

# 夏帕机器人（Sharpa Robotics）

| 字段 | 内容 |
|------|------|
| **机构** | 夏帕机器人（Sharpa Robotics），公司自称 AI robotics company |
| **成立** | 2024 年；新闻稿称 “Founded in 2024”（[PR Newswire 2025-12-16，经 Utusan 转载](https://www.utusan.com.my/pr-newswire-en/2025/12/ai-robotmaker-sharpa-reaches-key-milestone-with-mass-production-of-worlds-most-advanced-human-sized-robotic-hand/)） |
| **总部** | 新加坡（同一新闻稿）；第三方报道提及上海有制造与研发，未核实 |
| **旗舰产品** | SharpaWave 灵巧手（22 自由度，指尖触觉阵列）；新闻稿称 2025-12 开始量产 |
| **其他产品（文章）** | D01 整机、W02 灵巧手、AE01 触觉反馈数据手套（文章称于 IROS 2026 发布） |
| **官网** | [sharpa.com](https://www.sharpa.com/)（2026-10-16 首页核对，标题为 “Sharpa: AI robotics platform for dexterous manipulation”；成立年份与总部未在首页找到，以新闻稿为准） |
| **开源** | 公司级无统一开源入口；各论文的代码与数据状态见对应节点 |

## 一句话定义

**夏帕机器人**：以 **触觉灵巧手** 为核心硬件、以 **数据与接触控制** 为研究抓手的具身公司。学术负责人张凯峰（据文章）把灵巧操作的瓶颈归结为「接触」：数据缺力与触觉，底层控制要在接触瞬间高频反应。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 自由度；SharpaWave 为 22 DoF |
| VLA | Vision-Language-Action | 视觉–语言–动作模型，文中 System 1/2 的上层 |
| HOI | Hand–Object Interaction | 手–物交互，人类视频重建的核心对象 |
| EMG | Electromyography | 表面肌电，DexEMG 用其推断手部姿态 |
| Sim2Real | Simulation to Real | 仿真到真机迁移，多篇论文的共同路线 |

## 为什么重要

- **硬件是研究平台**：文中多篇论文（EgoScale、MoDE-VLA、T-Rex、Do as I Do、Dex-X 等）都以 Sharpa Wave 为真机，使不同团队的结果能在同一本体上比较。
- **数据两难的一个回应**：遥操作缺接触感受，人类视频缺力与触觉。公司路线是 **无本体采集**（AE01 手套 + 指尖触觉）与 **仿真补全接触**（Dex-X）并行。
- **分层架构**：文中描述 System 2（约 1 Hz，VLM）、System 1（约 10 Hz，运动）、System 0（约 100 Hz，实时触觉交互）。这是公司学术线的组织框架，不等于已公开的统一产品架构。

## 公开技术路线与时间

| 时间 | 事件 | 来源 |
|------|------|------|
| 2024 | 公司成立（据新闻稿） | PR Newswire 新闻稿 |
| 2025-10 | SaTA：空间锚定触觉感知（与清华合作） | 文章；[arXiv:2510.14647](https://arxiv.org/abs/2510.14647) |
| 2025-12-16 | SharpaWave 量产 | PR Newswire 新闻稿 |
| 2026 年初 | CES 2026 展示 CraftNet 分层模型等（文章称） | 文章 |
| 2026-08-29 | 与 Dairy Queen 合作的机器人餐厅在上海吴江路开业，55 步流程 | 文章（据报道 98% 步骤涉及触觉） |
| 2026-09（IROS 2026 期间） | 发布 D01 整机、W02 灵巧手、AE01 触觉手套 | 文章 |

### 三条研究主线（文章归纳）

1. **击穿具身鸿沟**：让人类数据成为动作燃料。代表：[UVTA](./paper-uvta-human-tactile-dexterous-manipulation.md)（无本体触觉采集）、[DexEMG](./paper-dexemg.md)（肌电遥操作）、[Dex-X](./paper-dex-x.md)（视频 + 仿真触觉补全）。
2. **解耦运动与交互**：接触交给仿真与原子技能。接口有三类：踏板式（[MoDE-VLA](./paper-sa-2603-08122-mode-vla-mixture-of-dexterous-experts-vla.md) 的 IMCopilot）、点云式（[Dex-X](./paper-dex-x.md)）、隐空间式（[TacBPM](./paper-tacbpm.md)）。
3. **以触觉为锚的跨本体表征**：代表 [FTP-1](./paper-ftp-1.md)（跨 21 种触觉传感器预训练）、[EgoScale](./paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div.md)、[T-Rex](./paper-trex-tactile-reactive-dexterous-manipulation.md)。

文中还列出与 Sharpa 硬件或合作相关的他队工作：[Do as I Do](./paper-do-as-i-do.md)、[SimToolReal](./simtoolreal.md)、[Play2Perfect](./paper-play2perfect.md)、[GALATEA](./paper-galatea.md)、[Tacmap](./paper-tacmap.md)。

## 2026-10-22 补充（天南文章）

三条路线的项目索引见 [Sharpa 学术路线阅读坐标](../overview/sharpa-academic-route-three-lines.md)；新增独立节点：[POISE](./paper-poise-in-hand-6d-pose-reaching.md)、[SharpaDex](./sharpadex.md)、[CraftNet](./craftnet.md)、[D01](./sharpa-d01-robot.md)、[W02](./sharpa-w02-hand.md)、[AE01](./sharpa-ae01-data-glove.md)。

## 工程实践

- 库内其他以 Sharpa 手为平台的论文：[SPD](./paper-spd.md)、[TeleDexter](./paper-teledexter.md)、[ArtManip](./paper-artmanip.md)、[WM-Craftnet](./paper-wm-craftnet.md)、[EigenDexplore](./paper-eigendexplore.md)（后两者 arXiv 与文章中的 CraftNet 不同，不等同）。
- 研究者视角的硬件要求（文章转述）：可靠性（“坏了之后一修又要两周”不可接受）与标定一致性，使仿真强化学习到真机的迁移更顺畅。

## 局限与风险

- **成本与可得性**：第三方报道给出 SharpaWave 约 5 万美元量级，未经公司核实；第一代产品据报道主要用于科研。
- **型号命名不一**：第三方页面出现 “Wave W01”、“North” 等名称，与文章中的 W02、D01 不一致；本页以文章为准。
- **论断性质**：文中关于 3–6 个月与约两年的时间判断，是学术负责人的个人判断，不是公开数据。
- **复现边界**：多篇论文依赖 Sharpa 硬件，复现需同类手或定制平台；各论文的开源状态以其节点为准。

## 关联页面

- 触觉与接触：[接触丰富操作](../concepts/contact-rich-manipulation.md)、[视触觉融合](../concepts/visuo-tactile-fusion.md)
- 论文节点（本次 ingest）：[SaTA](./paper-sata.md)、[Tacmap](./paper-tacmap.md)、[DexEMG](./paper-dexemg.md)、[FTP-1](./paper-ftp-1.md)、[Do as I Do](./paper-do-as-i-do.md)、[Play2Perfect](./paper-play2perfect.md)、[Dex-X](./paper-dex-x.md)、[TacBPM](./paper-tacbpm.md)、[UVTA](./paper-uvta-human-tactile-dexterous-manipulation.md)

## 参考来源

- [硬件、数据到模型：Sharpa如何系统性推进灵巧操作（机器之心，2026-10-16）](../../sources/blogs/wechat_sharpa_dexterous_manipulation_2026-10-16.md)
- [Sharpa 量产与成立时间（PR Newswire 新闻稿，2025-12-16，经 Utusan 转载）](https://www.utusan.com.my/pr-newswire-en/2025/12/ai-robotmaker-sharpa-reaches-key-milestone-with-mass-production-of-worlds-most-advanced-human-sized-robotic-hand/)
