---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2407.12381"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_flow-matching-imitation-learning-for-multi-suppo.md
summary: "Flow Matching Imitation Learning for Multi-Support Manipulation：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Flow Matching Imitation Learning for Multi-Support Manipulation

**Flow Matching Imitation Learning for Multi-Support Manipulation** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：14_Human_Motion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Flow Matching Imitation Learning for Multi-Support Manipulation 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
flowchart TB
    subgraph DEMO["示范采集（teleoperation 模式）"]
        OP["人类操作者"] --> TELE["遥操作：末端目标位姿<br/>+ 接触开关指令"]
        TELE --> DS["示范数据集<br/>末端指令 + marker 位姿"]
    end
    DS --> TRN["Flow Matching 训练<br/>flow 模型 f(z, t, s) 以状态 s 为条件"]
    subgraph POLY["高层控制（autonomous / shared autonomy）"]
        CAM["头部 RGB-D 相机<br/>AprilTags marker 位姿估计 100Hz"] --> S["策略输入状态 s<br/>末端位姿指令 · 接触状态<br/>marker 位姿 · 距上次切换 / 检测的时间"]
        S --> INF["Flow Matching 推理<br/>噪声轨迹经 flow 积分到目标轨迹"]
        INF --> OUT["输出：未来末端位姿轨迹<br/>+ 连续接触信号 γ"]
        OUT --> HYS["滞后阈值 → 离散接触开关"]
        OUT --> STI["轨迹拼接 · 线性插值<br/>5Hz 重采样到 100Hz · 低通滤波"]
        SH["shared autonomy：操作者指令<br/>替换其控制的那个末端"] --> STI
    end
    TRN --> INF
    HYS --> LOW
    STI --> LOW["SEIKO Retargeting + SEIKO Controller<br/>基于 SQP 的多接触全身控制 500Hz"]
    LOW --> ROB["Talos 人形机器人"]
    ROB -.-> CAM
```

图依据：论文 §III / Fig.2（整体架构）、§III-B/C/D/E。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/14_Human_Motion/flow-matching-imitation-learning-for-multi-suppo` |
| arXiv | <https://arxiv.org/abs/2407.12381> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页目前只能支撑一个结论：这是一条已进入 14_Human_Motion 深读队列、但尚未产出笔记的条目，它现在的价值只是可检索，而不是内容。**

- 页面能确证的信息仅限分类（14_Human_Motion）、arXiv 编号与计划文件夹路径；flow matching 如何用于多支撑操作、相对扩散策略的取舍如何，本页均无依据。
- 因此本页 **不可作为技术判断依据**：机制与指标应回到论文原文，或等 PROGRESS.md 追踪的深读笔记完成后再引用。
- 定位是 **待读条目**，方便你按分类找到它；笔记完成后应补成完整摘要，并重写本节为面向内容的结论。

## 与其他页面的关系

- 分类页：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_flow-matching-imitation-learning-for-multi-suppo.md](../../sources/papers/humanoid_pnb_flow-matching-imitation-learning-for-multi-suppo.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2407.12381>
- [arXiv PDF](https://arxiv.org/pdf/2407.12381)：已通读方法部分（实验与附录以略读为准）：§I–§III（Method 全部小节）；实验章节略读

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
