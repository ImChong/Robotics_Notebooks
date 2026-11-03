---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2506.00305"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-aerodynamics-for-the-control-of-flying.md
summary: "Learning Aerodynamics for the Control of Flying Humanoid Robots：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Learning Aerodynamics for the Control of Flying Humanoid Robots

**Learning Aerodynamics for the Control of Flying Humanoid Robots** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Learning Aerodynamics for the Control of Flying Humanoid Robots 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口，不给方法结论；论文方法的模块流程见下方「流程总览」。

## 流程总览

```mermaid
flowchart TD
    WT["风洞实验（PoliMi GVPM）<br/>iRonCub-Mk1 全尺寸，改变风速/攻角/侧滑角/关节构型<br/>测量整体气动力矩与表面压力"] --> VAL
    CFD1["与风洞同设置的 CFD<br/>稳态 RANS：Realizable k-ε 与 SST k-ω 对比"] --> VAL
    VAL["CFD 与风洞数据对比验证<br/>SST k-ω 更准确"] --> COARSE
    COARSE["网格敏感性分析<br/>选用约 0.33×10⁶ 单元的粗网格"] --> AUTO
    subgraph DATA["自动化 CFD 数据集生成"]
        AUTO["Ansys Workbench 随机生成关节构型<br/>Fluent Meshing 划网格<br/>pyFluent 自动旋转姿态并求解"] --> DS["约 8500 次仿真数据集<br/>24 种关节构型 × 19 俯仰角 × 18 偏航角<br/>对称镜像增广"]
    end
    DS --> DNN["DNN 模型<br/>输入：相对风向 + 19 关节自由度<br/>输出：各连杆气动力系数"]
    DS --> AXI["轴对称模型<br/>连杆视为孤立轴对称体<br/>Lasso + 最小二乘回归 C_D·A 与 C_N·A"]
    DNN --> SIM
    AXI --> SIM
    AXI --> CTRL
    subgraph CL["气动感知控制与验证"]
        CTRL["全身动量型飞行控制器<br/>质心动量方程加入气动力项<br/>推力矢量 + 相对阶增广"] --> SIM["Whole-Body Simulator（Simulink + iDynTree）<br/>气动力经雅可比转置汇总为广义气动力矩"]
        SIM --> REAL["iRonCub-Mk1 飞行仿真与真机平衡实验验证"]
    end
```

图依据：论文 §4.2–4.5 / §1 末段方法概述

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/learning-aerodynamics-for-the-control-of-flying` |
| arXiv | <https://arxiv.org/abs/2506.00305> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是「学习空气动力学以控制飞行人形」这项工作的阅读进度待读，不是方法归纳：页面里可确定的只有分类归属与论文入口，气动模型如何学、如何进控制器待深读笔记补齐。**

- 页面正文未给出任何方法、指标或消融，「实验与评测」已明确标注待笔记完成后补充。
- 适用边界：可用于按 05_Locomotion 分类检索与追踪阅读进度，不可当作该工作的结论来引用。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-aerodynamics-for-the-control-of-flying.md](../../sources/papers/humanoid_pnb_learning-aerodynamics-for-the-control-of-flying.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2506.00305>
- [arXiv PDF](https://arxiv.org/pdf/2506.00305)：已通读方法部分（实验与附录以略读为准）：§1 引言末段方法概述、§4 Methods（§4.1 机械设计、§4.2 风洞、§4.3 CFD 模型、§4.4 气动模型、§4.5 气动控制设计与仿真环境）；§2 结果略读，补充材料未读


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
