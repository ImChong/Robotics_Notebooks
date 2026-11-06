---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2604.07984"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_physics-based-motion-tracking-of-contact-rich-in.md
summary: "Physics-Based Motion Tracking of Contact-Rich Interacting Characters：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Physics-Based Motion Tracking of Contact-Rich Interacting Characters

**Physics-Based Motion Tracking of Contact-Rich Interacting Characters** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Physics-Based Motion Tracking of Contact-Rich Interacting Characters 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在深读笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」（依据论文方法部分）。

## 流程总览

```mermaid
flowchart TD
    A[参考动作片段数据集] --> B["动作采样：按近期平均跟踪奖励<br/>Softmax(-r_track / T) 调整片段采样概率"]
    B --> C["观测：人形状态 s_t（局部身体位置、线速度）<br/>+ 目标状态 g_t+1（位置/旋转/速度/关节偏差）"]
    C --> D["专家 π0：3 层 MLP<br/>先在全数据集上训练，其余专家冻结"]
    C --> G["Gating Network f_g<br/>Sigmoid 输出各专家置信度 r̃_k（预测奖励）"]
    D --> H["新专家 π_k 通过 lateral adapters<br/>读取已冻结专家的特征，学习残差动作偏移"]
    G --> I["Progressive Sampling<br/>β_k = max(1, r̃_k / r̃_k-1)<br/>预测奖励最低的样本路由给新专家"]
    I --> H
    D --> J["动作 = 各专家输出之和<br/>π_k = Σ π_i"]
    H --> J
    J --> K[动作作为 PD 控制器目标 → 关节力矩]
    K --> L[物理仿真环境：下一状态 + 跟踪奖励 + 能量惩罚]
    L --> B
    L --> N["PPO 更新：同一时刻只激活一个专家<br/>屏蔽冻结专家的 log σ，另加 adapter 使用损失"]
    N -.r̃_0 增长停滞时激活下一个专家.-> H
```

图依据：论文 §3.1、§3.2 / Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/13_Physics-Based_Animation/Physics-Based_Motion_Tracking_of_Contact-Rich_Interacting_Characters` |
| arXiv | <https://arxiv.org/abs/2604.07984> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页目前只是一条待读记录：论文已进入待深读队列，但「接触密集的交互角色动作跟踪」究竟怎么做、做到什么程度，本库尚无自己的判断。**

- 可确认的只有分类归属（13_Physics-Based_Animation）与 arXiv 出处；核心机制、消融与实机指标全部待深读补齐。
- 当前价值只是 **可检索**：这篇挂在分类页与总索引之下，方便你找到它。
- 深读笔记完成前，本页不应被当作结论来源；判断请以 PROGRESS.md 与论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_physics-based-motion-tracking-of-contact-rich-in.md](../../sources/papers/humanoid_pnb_physics-based-motion-tracking-of-contact-rich-in.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2604.07984>
- [arXiv PDF](https://arxiv.org/pdf/2604.07984)：已通读方法部分（实验与附录以略读为准）：§3.1、§3.2、Fig.2

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
