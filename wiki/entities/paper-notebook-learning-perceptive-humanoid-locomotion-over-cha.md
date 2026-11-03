---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2503.00692"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-perceptive-humanoid-locomotion-over-cha.md
summary: "Learning Perceptive Humanoid Locomotion over Challenging Terrain：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Learning Perceptive Humanoid Locomotion over Challenging Terrain

**Learning Perceptive Humanoid Locomotion over Challenging Terrain** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Learning Perceptive Humanoid Locomotion over Challenging Terrain 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在深读笔记写出来之前，这里只保留分类位置与原文入口，不给方法结论；论文方法部分的模块与数据流见下方「流程总览」。

## 流程总览

```mermaid
flowchart TD
    subgraph S1["第一阶段：Oracle Policy 训练"]
        PRIV["特权观测 o^p<br/>无噪 height map + 本体 + 接触力"] --> TENC["Terrain encoder"]
        TENC --> LSTM1["LSTM + MLP<br/>actor / critic"]
        LSTM1 --> PPO["PPO 训练<br/>仅正则化奖励，无预设动作参考"]
    end
    subgraph S2["第二阶段：Student Policy 蒸馏"]
        NOISY["带噪观测<br/>本体 + 带噪 height map + 指令"] --> BILSTM["Terrain encoder + BiLSTM"]
        BILSTM --> VIB["VAE / 变分信息瓶颈<br/>潜变量 z"]
        VIB --> DEC["解码器重建特权状态<br/>ELBO 损失"]
        VIB --> POL["Locomotion policy<br/>3 层全连接 + LayerNorm"]
        POL --> ACT["关节动作 a<br/>推理时用 μ 代替采样 z"]
    end
    PPO -->|"teacher 动作"| DAG["DAgger 模仿<br/>MSE 损失"]
    DAG --> POL
    DR["域随机化 + 高度图噪声模型<br/>(增益 / 形变 / GP 相关噪声)"] --> NOISY
    CUR["地形课程<br/>按行走距离调整难度"] -.-> PPO
    CUR -.-> DAG
    ACT --> DEP["真机部署<br/>雷达点云 + 里程计生成 height map<br/>ONNX Runtime"]
```

图依据：论文 §III-A Oracle Policy Training、§III-B Student Policy Training、§III-C Terrain Configuration and Curriculum、§IV-A Experimental Setup / Fig. 2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/learning-perceptive-humanoid-locomotion-over-cha` |
| arXiv | <https://arxiv.org/abs/2503.00692> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 05_Locomotion 下的待读条目，价值在于待读而非归纳：真正的技术内容要等深读笔记写完才会出现。**

- 页面可确证的只有分类归属、PROGRESS.md 上的「待深读」状态与 arXiv 编号 2503.00692；复杂地形感知行走的机制与实机指标均无记录。
- 适用边界：适合按分类页与总索引检索时的落点，不适合作为技术依据引用。
- 主要风险是被误读成已完成摘要；深读状态以上游 PROGRESS.md 为准，笔记完成后本页会重写。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-perceptive-humanoid-locomotion-over-cha.md](../../sources/papers/humanoid_pnb_learning-perceptive-humanoid-locomotion-over-cha.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2503.00692>
- [arXiv PDF](https://arxiv.org/pdf/2503.00692)：已通读方法部分（实验与附录以略读为准）：§III Methods（A–C）、§IV-A 实验设置与基线

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
