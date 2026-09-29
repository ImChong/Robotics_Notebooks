---
type: entity
tags: [paper, humanoid-paper-notebooks, teleoperation, manipulation, active-perception, mobile-manipulation, act, mit]
status: complete
updated: 2026-09-28
arxiv: "2411.00704"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../tasks/teleoperation.md
  - ../methods/action-chunking.md
  - ./paper-notebook-vision-in-action-learning-active-perception-from.md
  - ./paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md
  - ./paper-notebook-learning-to-look-seeking-information-for-decisio.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-to-look-around.md
summary: "本文提出一套集成 5 自由度（DOF）可动颈的遥操作系统，复刻自然人类头部运动与感知。系统支持窥视（peeking）、倾头（tilting）等行为，给操作者更好的环境视角、降低远程操作的认知负荷。作者在七个遥操作任务上展示收益，并研究可动颈如何通过增强空间感知、减少分布偏移（distribution shift）来改善模仿学习的自主策略训练——相比固定广角相机基线，可动颈在遥操作任务表现、操作者认知负荷与自主学习上都有改善。"
---

# Learning to Look Around

**Learning to Look Around: Enhancing Teleoperation and Learning with a Human-like Actuated Neck** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

本文提出一套集成 5 自由度（DOF）可动颈的遥操作系统，复刻自然人类头部运动与感知。系统支持窥视（peeking）、倾头（tilting）等行为，给操作者更好的环境视角、降低远程操作的认知负荷。作者在七个遥操作任务上展示收益，并研究可动颈如何通过增强空间感知、减少分布偏移（distribution shift）来改善模仿学习的自主策略训练——相比固定广角相机基线，可动颈在遥操作任务表现、操作者认知负荷与自主学习上都有改善。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Actuated Neck | 可动颈（5 DOF） |
| Peeking / Tilting | 窥视 / 倾头等头部行为 |
| Cognitive Load | 认知负荷 |
| Spatial Awareness | 空间感知 |
| Distribution Shift | 分布偏移 |
| Imitation Learning | 模仿学习 |

## 为什么重要

- **"会动的头"对遥操作与自主学习都有益**，与 ViA、EgoMI 主动视觉一脉；
- **减少分布偏移**是固定相机难做到的，可动颈天然缓解；
- **降低认知负荷**直接影响采集时长与数据质量；
- 对人形（本就有颈/头）是自然的硬件配置。

## 解决什么问题

固定相机限制遥操作与学习： - 看不全、需操作者**脑补**，**认知负荷高**； - 固定视角导致**分布偏移**，自主策略难学。

论文要：用**拟人可动颈**让"头会动"，改善遥操作体验与自主学习。

## 核心机制

1. **5-DOF 拟人可动颈遥操作系统**：窥视/倾头等自然头动；
2. **降低操作者认知负荷**：七个任务展示收益；
3. **改善模仿学习**：增强空间感知、减少分布偏移；
4. **对照固定广角相机**：可动颈全面更优。

方法拆解（深读笔记小节）：5-DOF 拟人可动颈；降低遥操作认知负荷；改善模仿学习（空间感知 + 减分布偏移）；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_to_Look_Around__Enhancing_Teleoperation_and_Learning/Learning_to_Look_Around__Enhancing_Teleoperation_and_Learning.html> |
| arXiv | <https://arxiv.org/abs/2411.00704> |
| 源码 | **未开源**：论文脚注称 “The teleoperation code will be released”，截至 2026-09-28 未见官方仓库；正文引用的 [Improbable-AI/VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) 是同实验室的通用 Vision Pro 流式库，不是本文系统代码 |
| 作者 | Bipasha Sen、Michelle Wang、Nandini Thakur、Aditya Agarwal、Pulkit Agrawal（MIT） |
| 发表 | 2024 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**平台**：Husarion Panther 移动底盘 + 双 UR5e（平行夹爪，右手可换 Psyonic Ability 灵巧手）+ Interbotix WidowX-200 充当 5-DoF 颈部，共 21 DoF（另 6 DoF 灵巧手）；4 个 RealSense D405（颈、躯干、双腕）。头部追踪用 Apple Vision Pro（颈部相机画面经 WiFi 回传），手部用 trakSTAR 电磁追踪 / Manus 手套。

- **遥操作**：7 个全身任务（装洗碗机、挂卫衣、接水、摆餐桌、煮咖啡、从冰箱取物、扔垃圾）以定性演示为主，论文未给出量化用户研究。尝试过用 Vision Pro 做手部追踪，但近身任务中手会出视野，导致机械臂抖动，存在安全隐患。
- **自主学习**：ACT 只控制右臂 + 颈部（左臂固定，上下文窗口 1），3 个任务各 120 条演示合并训练单一策略（不给任务标签）；对照组用 160° 超广角静态相机，同样 120 条 / 任务。每任务 15 次试验，物体位置略作变动。

| 任务 | 可动颈部 | 静态超广角 |
|------|---:|---:|
| CfB：从茶几下取杯放入高处盒子 | **95%** | 0% |
| L2Rmod：左侧取物放上层架（去掉「探头」环节） | **90%** | 78% |
| CRange：近身物体操作 | **82%** | 68% |

- **分工**：颈部相机负责全局定位（如找到桌下的杯子），腕部相机负责抓取对准与释放时机。
- **静态相机失败原因**：CfB 中杯子只出现在全局画面底部，腕部相机卡在桌沿；作者认为广角相机对细小场景变化（如地上电线位置）更敏感，分布偏移更大。

## 与其他工作对比

| 工作 | 视角来源 | 与本文的差异 |
|------|------|------|
| 静态超广角相机（论文基线） | 固定 160° 视野 | 无法绕开遮挡，CfB 完全失败 |
| [Vision in Action](./paper-notebook-vision-in-action-learning-active-perception-from.md) | 6-DoF 颈部 + 点云渲染 VR | 用异步点云渲染降低眩晕；本文直接回传颈部相机视频 |
| [EgoMI](./paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md) | 人类 VR 采集 + 头部重定向 | 采集无需机器人；本文为机器人在环遥操作 |
| [Learning to Look](./paper-notebook-learning-to-look-seeking-information-for-decisio.md) | 学习型信息寻求策略 | 决定「何时找信息」；本文的颈部动作由模仿人类遥操作得来 |

## 结论

**这篇工作的主张是：遥操作的瓶颈有一部分不在手上而在头上——给系统一个 5 自由度拟人可动颈，同时改善了人操作时的体验与之后自主策略的学习质量。**

- 起作用的机制是让视角本身成为可控自由度：窥视、倾头这类自然头动替代了操作者的脑补，直接压低认知负荷。
- 对自主学习的收益走的是另一条通路：可动颈增强空间感知并减少分布偏移，而分布偏移正是固定相机难以缓解的问题。
- 证据形态分两层：七个遥操作任务是定性展示；与静态超广角相机的量化对照只在三个自主学习任务上（95% vs 0%、90% vs 78%、82% vs 68%）。
- 适用边界：方案建立在本就有颈/头结构的平台上是自然配置，换到没有可动头部的硬件则需先做改装。
- 工程含义落在采集侧：若认知负荷确实下降，会直接影响采集时长与数据质量——但论文没有给出用户研究数据，这一点仍待验证。

## 局限与风险

- **遥操作收益缺乏量化**：7 个遥操作任务是定性展示，「降低认知负荷」没有用户研究数据支撑。
- **自主实验范围小**：只控右臂 + 颈部、左臂固定；3 个任务、每任务 15 次；上下文窗口 1，不涉及长时序推理。
- **L2R 对比经过修改**：为公平比较去掉了「探头」环节（静态相机做不到），因此该任务低估了可动颈部的优势。
- **Vision Pro 手部追踪不可靠**：近身操作时手出视野导致抖动，论文改用电磁追踪与数据手套。
- **开源边界**：遥操作代码称将发布但未见；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 任务语境：遥操作：[teleoperation](../tasks/teleoperation.md)
- 自主策略所用 ACT：[action-chunking](../methods/action-chunking.md)
- 6-DoF 颈部 + 点云 VR 的后续对照：[paper-notebook-vision-in-action-learning-active-perception-from](./paper-notebook-vision-in-action-learning-active-perception-from.md)
- 无机器人采集的主动视觉：[paper-notebook-egomi-learning-active-vision-and-whole-body-mani](./paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md)
- 决策层信息寻求：[paper-notebook-learning-to-look-seeking-information-for-decisio](./paper-notebook-learning-to-look-seeking-information-for-decisio.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 遥操作主动视角在感知栈选型中的位置

## 参考来源

- [humanoid_pnb_learning-to-look-around.md](../../sources/papers/humanoid_pnb_learning-to-look-around.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_to_Look_Around__Enhancing_Teleoperation_and_Learning/Learning_to_Look_Around__Enhancing_Teleoperation_and_Learning.html>
- 论文：<https://arxiv.org/abs/2411.00704>
- 论文正文（硬件、实验与 Table 1）：<https://arxiv.org/html/2411.00704>

## 推荐继续阅读

- [机器人论文阅读笔记：Learning to Look Around](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_to_Look_Around__Enhancing_Teleoperation_and_Learning/Learning_to_Look_Around__Enhancing_Teleoperation_and_Learning.html)
- Vision Pro 流式库（同实验室通用工具）：<https://github.com/Improbable-AI/VisionProTeleop>
