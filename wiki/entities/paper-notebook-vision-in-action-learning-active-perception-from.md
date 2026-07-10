---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2506.15666"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_vision-in-action.md
summary: "Vision in Action（ViA）是面向双臂机器人操作的主动感知系统，直接从人类演示学任务相关的主动感知策略（如搜索、跟踪、聚焦）。硬件上，ViA 用一个简单有效的 6 自由度机器人颈实现灵活、拟人的头部运动。为捕捉人类主动感知策略，设计了基于 VR 的遥操作接口，在机器人与操作者之间建立共享观测空间。为缓解机器人物理运动延迟导致的VR 眩晕，接口用中间 3D 场景表征，在操作者端实时渲染视角、并异步用机器人最新观测更新场景。这些设计共同支撑了在三个含视觉遮挡的复杂多阶段双臂操作任务上学到鲁棒视觉运动策略，显著优于基线。"
---

# Vision in Action

**Vision in Action: Learning Active Perception from Human Demonstrations** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

Vision in Action（ViA）是面向双臂机器人操作的主动感知系统，直接从人类演示学任务相关的主动感知策略（如搜索、跟踪、聚焦）。硬件上，ViA 用一个简单有效的 6 自由度机器人颈实现灵活、拟人的头部运动。为捕捉人类主动感知策略，设计了基于 VR 的遥操作接口，在机器人与操作者之间建立共享观测空间。为缓解机器人物理运动延迟导致的VR 眩晕，接口用中间 3D 场景表征，在操作者端实时渲染视角、并异步用机器人最新观测更新场景。这些设计共同支撑了在三个含视觉遮挡的复杂多阶段双臂操作任务上学到鲁棒视觉运动策略，显著优于基线。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| ViA | Vision in Action，主动感知系统 |
| Active Perception | 主动感知（搜索/跟踪/聚焦） |
| 6-DoF Neck | 6 自由度机器人颈 |
| Shared Observation Space | 人机共享观测空间 |
| 3D Scene Representation | 中间 3D 场景表征（缓解延迟/眩晕） |
| Asynchronous Update | 异步更新 |

## 为什么重要

- **主动感知（会动的头）对遮挡任务是刚需**，固定相机看不全；
- **共享观测空间 + 3D 表征**是高质量遥操作采集的关键工程；
- **缓解 VR 眩晕**直接影响数据质量与采集时长；
- 与 EgoMI（头手协调）共同强调"主动视觉"对操作的价值。

## 解决什么问题

双臂操作中**视觉遮挡**常见，需**主动调整视角**： - 固定相机看不全，需**主动搜索/跟踪/聚焦**； - 想**从人类演示学**主动感知，但遥操作有**延迟**致**VR 眩晕**； - 缺**拟人头部硬件**与**共享观测**接口。

ViA 要：硬件（6-DoF 颈）+ 接口（VR 共享观测 + 3D 表征缓延迟）+ 从人类演示学主动感知。

## 核心机制

1. **主动感知系统 ViA**：从人类演示学搜索/跟踪/聚焦；
2. **6-DoF 机器人颈**：灵活拟人头部运动；
3. **VR 共享观测 + 3D 表征**：采集主动感知并缓解延迟/眩晕；
4. **遮挡任务显著领先**：三个多阶段双臂任务优于基线。

方法拆解（深读笔记小节）：6 自由度机器人颈（拟人头动）；VR 遥操作 + 共享观测空间；3D 场景表征缓解延迟/眩晕；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Vision_in_Action__Learning_Active_Perception_from_Human_Demonstrations/Vision_in_Action__Learning_Active_Perception_from_Human_Demonstrations.html> |
| arXiv | <https://arxiv.org/abs/2506.15666> |
| 作者 | Haoyu Xiong、Xiaomeng Xu、Jimmy Wu、Yifan Hou、Jeannette Bohg、Shuran Song（Stanford） |
| 发表 | 2025 年 6 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_vision-in-action.md](../../sources/papers/humanoid_pnb_vision-in-action.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Vision_in_Action__Learning_Active_Perception_from_Human_Demonstrations/Vision_in_Action__Learning_Active_Perception_from_Human_Demonstrations.html>
- 论文：<https://arxiv.org/abs/2506.15666>

## 推荐继续阅读

- [机器人论文阅读笔记：Vision in Action](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Vision_in_Action__Learning_Active_Perception_from_Human_Demonstrations/Vision_in_Action__Learning_Active_Perception_from_Human_Demonstrations.html)
