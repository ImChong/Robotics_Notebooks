---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2506.12314"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_explosive-output-to-enhance-jumping-ability-a-va.md
summary: "Explosive Output to Enhance Jumping Ability：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Explosive Output to Enhance Jumping Ability

**Explosive Output to Enhance Jumping Ability: A Variable Reduction Ratio Design Paradigm for Humanoid Robots Knee Joint** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：12_Hardware_Design）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Explosive Output to Enhance Jumping Ability 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 深读笔记写出来之前，这里只保留分类位置与原文入口；方法（设计）流程见下方「流程总览」，仅依据论文方法部分，不含结论判断。

## 流程总览

```mermaid
flowchart TD
    A["电机输出特性分析<br/>转矩 / 功率包络，避开高功率损耗区，宜工作在中速区"] --> C["EVRR-K 策略：减速比随膝关节伸展逐步降低<br/>起跳初期高比值快速建立转矩，之后降低以抑制电机转速增长"]
    B["简化腿模型的膝-质心运动学分析<br/>传动比 λ(q2) = 1 / J_CoM,y(q2)"] --> C
    C --> M["机构实现：直线执行器驱动的导杆机构<br/>双导程滚珠丝杠，减速比 k 随关节角变化"]
    M --> P["可调结构参数：曲柄长 r、机架长 S0、角度偏置 Δθ"]
    P --> O["基于爆发式跳跃控制的参数优化<br/>电机以最大可用转矩工作，最大化起跳能量 W_takeoff<br/>并与最优固定减速比（FRR-K）对比"]
    O --> V1["1-DOF 腿式平台垂直跳实验"]
    O --> V2["全尺寸人形 BHR8-J1 跳跃实验"]
```

图依据：论文 §II-§V（Fig. 3、Fig. 4、Fig. 6、Algorithm 1）；本文为机构设计范式，流程图为设计流程而非学习算法。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/explosive-output-to-enhance-jumping-ability-a-va` |
| arXiv | <https://arxiv.org/abs/2506.12314> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Explosive Output to Enhance Jumping Ability 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2506.12314、分类 12_Hardware_Design、计划中的笔记文件夹；论文的方法机制、评测口径与实机结果在本页一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-12-hardware-design.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_explosive-output-to-enhance-jumping-ability-a-va.md](../../sources/papers/humanoid_pnb_explosive-output-to-enhance-jumping-ability-a-va.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2506.12314>
- [arXiv PDF](https://arxiv.org/pdf/2506.12314)：已通读方法部分（实验与附录以略读为准）：§II、§III、§IV、§V 开头（实验设置）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
