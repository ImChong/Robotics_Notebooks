---

type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned, berkeley]
status: planned
updated: 2026-09-15
arxiv: "2407.21781"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_berkeley-humanoid-a-research-platform-for-learni.md
summary: "Berkeley Humanoid：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Berkeley Humanoid

**Berkeley Humanoid: A Research Platform for Learning-based Control** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：12_Hardware_Design）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Berkeley Humanoid 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见上方「流程总览」（依据论文方法部分），不含结论性判断。

## 流程总览

```mermaid
flowchart LR
    A["CAD 设计<br/>自研轻量人形硬件"] --> B["Isaac Lab 仿真训练<br/>PPO"]
    DR["设计驱动的精确域随机化<br/>CAD 惯量 + 执行器摩擦测量<br/>接触：摩擦 · 恢复系数 · 外力扰动"] --> B
    subgraph POL["最小化 RL 控制器（§4.1）"]
        OBS["观测：base 角速度 · 投影重力<br/>关节位置 / 速度 · 状态估计器的 base 线速度<br/>速度指令 · 上一步动作"] --> NET["MLP actor-critic<br/>隐层 512-256-128 · ELU"]
        NET --> ACT["动作：期望关节位置 qd"]
    end
    B --> POL
    ACT --> PD["电机驱动器上的 PD 控制器<br/>转换为力矩"]
    PD --> HW["Berkeley Humanoid 实机<br/>零样本 sim-to-real 部署"]
```

图依据：论文 Fig.1 与 §4（Minimally Composed Learning-based Controller）。此外，该论文主体是硬件平台，上图只覆盖其学习控制与部署链路，不含硬件设计细节。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/berkeley-humanoid-a-research-platform-for-learni` |
| arXiv | <https://arxiv.org/abs/2407.21781> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前提供的是「待读条目」而不是论文解读：它把 Berkeley Humanoid 这条 12_Hardware_Design 条目挂进知识库以保持检索连通，实质技术判断要等深读笔记落地。**

- 页面已确认的信息仅有分类（12_Hardware_Design）、arXiv 编号与计划文件夹路径；这套面向学习式控制的研究平台的硬件设计与量化指标均尚未记录。
- 适用边界很窄：可用于按分类检索，不能作为该论文技术结论的引用依据。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_berkeley-humanoid-a-research-platform-for-learni.md](../../sources/papers/humanoid_pnb_berkeley-humanoid-a-research-platform-for-learni.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2407.21781>
- [arXiv PDF](https://arxiv.org/pdf/2407.21781)：已通读方法部分（实验与附录以略读为准）：Fig.1、§4（RL 控制器与 sim-to-real 设计）；§3 硬件设计与 §5 实验仅略读

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
