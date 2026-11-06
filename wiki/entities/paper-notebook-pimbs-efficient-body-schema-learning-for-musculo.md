---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2506.20343"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_pimbs-efficient-body-schema-learning-for-musculo.md
summary: "PIMBS：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# PIMBS

**PIMBS: Efficient Body Schema Learning for Musculoskeletal Humanoids** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：12_Hardware_Design）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

PIMBS 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 深读笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」，仅依据论文方法部分，不含结论判断。

## 流程总览

```mermaid
flowchart TD
    DATA["实机运动采集数据<br/>D_i = (θ, f, l)<br/>随机肌张力 或 几何模型指令肌长 + 张力控制消除松弛"] --> NET["神经网络 h：输入 θ（AL-Map）或 (θ, f)（ATL-Map）<br/>输出肌肉长度 l_pred"]
    NET --> LB["L_basic：l_pred 与 l_data 的均方误差"]
    NET --> DIFF["对输入 θ 求导得肌肉雅可比 G_pred"]
    DIFF --> LP["L_pinn：G_pred^T f + τ_data 的平方误差<br/>τ 为假定关节结构正确时的重力补偿力矩，τ = -G^T f"]
    NET --> LC["L_const：θ=0, f=0 时 l 应为 0"]
    LB --> LOSS["总损失 L_basic + L_const + α·L_pinn"]
    LC --> LOSS
    LP --> LOSS
    LOSS -->|"反向传播更新权重 W"| NET
    NET --> USE["训练后的身体图式 h：<br/>输出期望肌长；G 用于状态估计与力矩控制"]
```

图依据：论文 §II-B、§II-C（Fig. 3、Fig. 4，Eq. 6-12）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/pimbs-efficient-body-schema-learning-for-musculo` |
| arXiv | <https://arxiv.org/abs/2506.20343> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 PIMBS 的待读条目，价值在于让肌骨人形（12_Hardware_Design）这一分支不漏掉这一篇，而不是给出 body schema 学习的结论。**

- 页面上唯一确定的信息是归类与计划路径：分类 12_Hardware_Design、计划文件夹 `papers/12_Hardware_Design/pimbs-efficient-body-schema-learning-for-musculo`，深读状态仍为「待撰写」。
- 适用边界：可用于分类检索，不可当作方法或指标依据——本页未给出任何机制描述与量化结果。
- 主要风险是被误当成已消化的笔记：本页「实验与评测」已明确 benchmark、消融与实机指标待补，引用前应回到 PROGRESS.md 与 <https://arxiv.org/abs/2506.20343>。

## 与其他页面的关系

- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_pimbs-efficient-body-schema-learning-for-musculo.md](../../sources/papers/humanoid_pnb_pimbs-efficient-body-schema-learning-for-musculo.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2506.20343>
- [arXiv PDF](https://arxiv.org/pdf/2506.20343)：已通读方法部分（实验与附录以略读为准）：§II-A 至 §II-C、§III-A

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
