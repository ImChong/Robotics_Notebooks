---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-10-06
arxiv: "2502.10894"
related:
  - ../overview/paper-notebook-category-10-sim-to-real.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
sources:
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/papers/humanoid_pnb_bridging-the-sim-to-real-gap-for-athletic-loco-m.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation

**Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：10_Sim-to-Real）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在深读笔记写出来之前，这里只保留分类位置、原文入口与论文方法部分的流程总览，不给实验结论。

## 流程总览

```mermaid
flowchart TD
  subgraph S1["阶段 1: 真实到仿真校准 (§II-A)"]
    HW[真机采集转移数据 s_t, τ_t, s_t+1<br/>方波 / 正弦波 / 高斯噪声激励] --> UANT[用 RL 训练无监督执行器网络 UAN<br/>输入 20 步位置与速度误差历史, 输出修正力矩 δτ]
    UANT --> UANR[奖励: 仿真与真实关节位置差 + 平滑项]
  end
  subgraph S2["阶段 2: 全身控制器 WBC 训练"]
    PRE[预训练 §II-B: Isaac Sim 4096 并行环境 + PPO<br/>跟踪随机基座速度与末端位姿指令, 域随机化]
    PRE --> FT[任务微调 §II-C: 沿用预训练权重<br/>读入任务参考轨迹 + one-hot 任务嵌入, 用任务奖励训练]
    FT --> DEV[策略逐步偏离参考轨迹以提升任务表现]
  end
  UANT -->|UAN 接入仿真环路| PRE
  UANT -->|UAN 接入仿真环路| FT
  DEV --> DEP[阶段 3: 部署到 Unitree B2 + Z1 Pro 真机]
```

图依据：论文 §II Method / Fig.2（三步流程：训练 UAN、预训练并微调 WBC、部署）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 10_Sim-to-Real |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/10_Sim-to-Real/bridging-the-sim-to-real-gap-for-athletic-loco-m` |
| arXiv | <https://arxiv.org/abs/2502.10894> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前是一条待读记录，而不是对这篇论文的归纳——它记录的是「这篇该读、归在哪一类、还没读」。**

- 页面能给出的确定信息只有四项：分类为 10_Sim-to-Real、深读状态为待撰写、计划文件夹路径已预留、arXiv 入口已登记。
- 论文方法流程见上方「流程总览」；量化 benchmark 与实机结论 **本页均未涵盖**，不要把这里的任何表述当作对该工作的评价。
- 它的作用是让你能从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到它到该实体；深读笔记完成后本页应链向笔记站并补成完整摘要。

## 与其他页面的关系

- 分类页：[paper-notebook-category-10-sim-to-real](../overview/paper-notebook-category-10-sim-to-real.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_bridging-the-sim-to-real-gap-for-athletic-loco-m.md](../../sources/papers/humanoid_pnb_bridging-the-sim-to-real-gap-for-athletic-loco-m.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2502.10894>
- [arXiv PDF](https://arxiv.org/pdf/2502.10894)：已通读方法部分（实验与附录以略读为准）：§II Method（II-A 至 II-C）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
