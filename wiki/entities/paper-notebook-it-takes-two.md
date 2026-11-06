---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2510.10206"
venue: "2025.10"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_it-takes-two.md
summary: "It Takes Two：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# It Takes Two

**It Takes Two: Learning Interactive Whole-Body Control Between Humanoid Robots** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

It Takes Two 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」，效果结论仍待深读。

## 流程总览

```mermaid
flowchart TD
    IN["两人交互动捕序列（SMPL 格式）"] --> R1
    subgraph RT["阶段一：接触感知运动重定向"]
        R1["Trimesh 网格碰撞检测：得到接触面对 C1:T"] --> R2["带 β 正则的形状优化（L_keypoint + λ‖β'‖²）"]
        R2 --> R3["接触顶点映射到机器人网格"]
        R3 --> R4["优化第一个体的根位姿偏移 Δp_root, Δθ_root"]
        R4 --> R5["关节 L2 对齐重定向 → 机器人参考运动 + 接触掩码"]
    end
    R5 --> C0
    subgraph CT["阶段二：交互驱动运动控制器（RL，两机器人共享策略）"]
        C0["观测：本体感觉 + 参考目标 + 对方状态摘要 + 参考/实测接触"] --> C1["策略输出目标关节角 → PD 控制"]
        C1 --> C2["奖励：跟踪 + 交互感知全局奖励 + 接触奖励"]
        C2 --> C3["在线课程：早期重跟踪，逐步加大交互与接触权重"]
    end
    C3 --> OUT["两台人形机器人协同交互动作"]
```

图依据：论文 §4 / Fig. 2、Fig. 3

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/It_Takes_Two__Learning_Interactive_Whole-Body_Control_Between_Humanoid_Robots` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 It Takes Two 的阅读进度待读，不是方法归纳：页面里可确定的只有分类归属与计划文件夹，人形之间的交互式全身控制如何学待深读笔记补齐。**

- 页面正文未给出任何方法、指标或消融，「实验与评测」已明确标注待笔记完成后补充。
- 适用边界：可用于按 04_Loco-Manipulation_and_WBC 分类检索与追踪阅读进度，不可当作该工作的结论来引用。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_it-takes-two.md](../../sources/papers/humanoid_pnb_it-takes-two.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2510.10206)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 预备知识（含孤立问题）、§4.1 接触感知运动重定向、§4.2 交互驱动运动控制器（观测、奖励、课程）；§5 实验与附录仅略读或未读。

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
