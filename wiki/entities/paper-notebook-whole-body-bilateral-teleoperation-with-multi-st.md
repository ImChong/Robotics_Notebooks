---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2508.09846"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_whole-body-bilateral-teleoperation-with-multi-st.md
summary: "本文提出一个面向轮式人形移动操作的物体感知全身双边遥操作（object-aware whole-body bilateral teleoperation）框架，把遥操作与在线参数估计结合。它顺序整合：① 基于视觉的物体尺寸估计；② 由大型视觉语言模型（VLM）生成的初始参数猜测；③ 一个解耦的分层采样策略（先估质量/质心、再推惯量）。估出的参数用于实时更新机器人的平衡点（equilibrium point）。在搬运约机器人体重 1/3 的负载（抬、送、放）时，框架实现更动态的全身遥操作，同时保持柔顺，并通过物体动力学补偿改善操作跟踪；在自制轮式人形 + 夹爪上实时验证。"
---

# Whole-Body Bilateral Teleoperation with Multi-Stage Object Parameter Estimation for Wheeled Humanoid Locomanipulation

**Whole-Body Bilateral Teleoperation with Multi-Stage Object Parameter Estimation for Wheeled Humanoid Locomanipulation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

本文提出一个面向轮式人形移动操作的物体感知全身双边遥操作（object-aware whole-body bilateral teleoperation）框架，把遥操作与在线参数估计结合。它顺序整合：① 基于视觉的物体尺寸估计；② 由大型视觉语言模型（VLM）生成的初始参数猜测；③ 一个解耦的分层采样策略（先估质量/质心、再推惯量）。估出的参数用于实时更新机器人的平衡点（equilibrium point）。在搬运约机器人体重 1/3 的负载（抬、送、放）时，框架实现更动态的全身遥操作，同时保持柔顺，并通过物体动力学补偿改善操作跟踪；在自制轮式人形 + 夹爪上实时验证。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Bilateral Teleop | 双边遥操作，力/位双向反馈 |
| Object-Aware | 物体感知，估计被操作物体参数 |
| VLM | Vision-Language Model，视觉语言模型 |
| Equilibrium Point | 平衡点，控制中维持平衡的参考 |
| CoM / Inertia | 质心 / 惯量，物体动力学参数 |
| Wheeled Humanoid | 轮式人形 |

## 为什么重要

- **遥操作 + 在线辨识**是搬未知重物的务实组合：估出物性才能稳；
- **VLM 给物理参数先验**是新颖用法，把语义视觉接到动力学估计；
- **分层估计（先质量/质心再惯量）**降低辨识难度；
- 与 SplitAdapter、Heavy-Lifting 等负载/重物工作呼应，强调"物体动力学"建模。

## 解决什么问题

轮式人形搬运未知重物时： - 物体的**质量/质心/惯量未知**，会扰动全身平衡； - 遥操作若不补偿物体动力学，操作**跟踪差、易失稳**。

论文要：**在线估计物体参数**并补偿，让双边遥操作在搬重物时**动态、柔顺、跟踪准**。

## 核心机制

1. **物体感知双边遥操作**：把在线物体参数估计纳入全身遥操作；
2. **多阶段估计**：视觉尺寸 → VLM 先验 → 分层采样（质量/质心→惯量）多假设；
3. **平衡点实时补偿**：用估出的参数补偿负载对全身平衡的影响；
4. **重载验证**：约 1/3 自重负载下更动态、柔顺、跟踪更好。

方法拆解（深读笔记小节）：多阶段物体参数估计；实时更新平衡点；双边遥操作 + 并行仿真/硬件；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation.html> |
| arXiv | <https://arxiv.org/abs/2508.09846> |
| 作者 | Donghoon Baek、Amartya Purushottam、Jason J. Choi、Joao Ramos（UIUC） |
| 发表 | 2025 年 8 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_whole-body-bilateral-teleoperation-with-multi-st.md](../../sources/papers/humanoid_pnb_whole-body-bilateral-teleoperation-with-multi-st.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation.html>
- 论文：<https://arxiv.org/abs/2508.09846>

## 推荐继续阅读

- [机器人论文阅读笔记：Whole-Body Bilateral Teleoperation with Multi-Stage Object Parameter Estimation for Wheeled Humanoid Locomanipulation](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation.html)
