---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2508.19926"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_farm-frame-accelerated-augmentation-and-residual.md
summary: "FARM：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# FARM

**FARM: Frame-Accelerated Augmentation and Residual Mixture-of-Experts for Physics-Based High-Dynamic Humanoid Control** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

FARM 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    AM["AMASS 数据集"] --> ACC["整体加速 1.25×"]
    ACC --> FC["基线 FC 控制器推理跟踪"]
    FC --> HARD["跟踪失败样本（全局平均关节位置误差在任一帧 > 0.5 m）= 困难样本"]
    HARD --> AUG["帧加速增广：加速因子 v ~ U[1.0, 1.5]，线性插值重采样"]
    AUG --> TRAIN["目标条件 RL（PPO，Isaac Lab）训练 FARM 策略"]
    subgraph NET["残差 MoE 结构"]
        IN["冻结的 FC 输入 MLP"] --> TF["冻结的 FC Transformer → z_base"]
        IN --> ROUTER["速度感知路由 SAR（按参考帧平均关节速度分 3 档，交叉熵监督）"]
        ROUTER --> DEA["动态专家分配 DEA：k = argmax(p)，激活专家 1…k"]
        DEA --> RES["残差 Δz（专家复制 FC Transformer + 零初始化线性层）"]
        TF --> SUM["z = z_base + Δz"]
        RES --> SUM
        SUM --> OUT["可训练的输出 MLP → PD 目标关节角"]
    end
    TRAIN --> NET
    OUT --> EVAL["在 HDHM 高动态动作集上评估泛化"]
```

图依据：论文 §3.1–§3.3 / Fig. 2、Fig. 3 与式 (1)–(4)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/13_Physics-Based_Animation/farm-frame-accelerated-augmentation-and-residual` |
| arXiv | <https://arxiv.org/abs/2508.19926> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**FARM 在本库目前只是一条「已排期、未深读」的待读条目：它只标出这篇落在物理动画分支下"高动态人形控制"这条线这条线上，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2508.19926、分类 13_Physics-Based_Animation 与计划中的笔记文件夹；标题指向"帧加速增强 + 残差专家混合用于基于物理的高动态人形控制"，两个组件各自解决什么、如何组合，页内一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-13-physics-based-animation.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为方法对比或复现依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 PROGRESS.md 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_farm-frame-accelerated-augmentation-and-residual.md](../../sources/papers/humanoid_pnb_farm-frame-accelerated-augmentation-and-residual.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2508.19926>
- [arXiv PDF](https://arxiv.org/pdf/2508.19926)：已通读方法部分（实验与附录以略读为准）：§3 Method（3.1–3.3）

## 推荐继续阅读

- [机器人论文阅读笔记：FARM](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/FARM__Frame-Accelerated_Augmentation_and_Residual_MoE_for_High-Dynamic_Humanoid/FARM__Frame-Accelerated_Augmentation_and_Residual_MoE_for_High-Dynamic_Humanoid.html)
- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
