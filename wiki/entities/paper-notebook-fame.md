---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2603.08961"
venue: "2026.03"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_fame.md
summary: "FAME：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# FAME

**FAME: Force-Adaptive RL for Expanding the Manipulation Envelope of a Full-Scale Humanoid** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

FAME 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口，不给方法结论。

## 流程总览

```mermaid
flowchart TD
    subgraph TR["训练（仿真）"]
        PC["III-A 上半身姿态课程 (HOMIE)<br/>站立质量提升 → 扩大 q_ub 随机目标范围<br/>每 1 s 重采样"] --> Q["上半身关节位置 q_ub ∈ R15"]
        FS["III-B 双手施力：各向同性 3D 随机力 F_L, F_R"] --> ENC
        Q --> ENC["上半身 Context Encoder (MLP)<br/>z_t = μ([q_ub, F_L, F_R]) ∈ R8<br/>与策略端到端训练，无力标签"]
        ENC --> OBS["观测 o_t：3 步本体感觉历史 + 3 步 z 历史"]
        OBS --> POL["III-D 基础站立策略 π (PPO, RSL-RL)<br/>输出 12D 下肢关节目标偏置"]
        POL --> PD["50 Hz 关节 PD 控制"]
    end
    subgraph DP["部署（H1-2，无腕部力传感）"]
        JT["手臂关节力矩 / 状态"] --> EST["III-C 手部力估计<br/>单臂 7 自由度动力学行 + RNEA (Pinocchio)<br/>最小二乘解 F̂"]
        EST --> ENC2["同一 Context Encoder<br/>（用估计力代替真实力）"]
        ENC2 --> POL2["站立策略 → 下肢 PD 目标"]
    end
    TR -.->|"训练得到的策略与编码器"| DP
```

图依据：论文 §III（Fig. 1 / Fig. 2；III-A 姿态课程、III-B 编码器与施力、III-C 手部力估计、III-D 基础策略）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/FAME__Force-Adaptive_RL_for_Expanding_the_Manipulation_Envelope_of_a_Full-Scale_` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**FAME 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2603.08961（2026.03）、分类 04_Loco-Manipulation_and_WBC、计划中的笔记文件夹；论文的方法机制、评测口径与实机结果在本页一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_fame.md](../../sources/papers/humanoid_pnb_fame.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2603.08961)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§III 方法（A–E）、§IV 实验协议与 Table II 略读


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
