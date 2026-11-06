---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2505.06584"
venue: "2025.05"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_jaeger.md
summary: "JAEGER：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# JAEGER

**JAEGER: Dual-Level Humanoid Whole-Body Controller** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

JAEGER 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在深读笔记写出来之前，这里只保留分类位置与原文入口；方法流程见「流程总览」（依据论文方法部分），结论与指标以论文为准。

## 流程总览

```mermaid
flowchart TD
    subgraph RT["I 重定向网络 Retargeting Network"]
        A["AMASS 数据集"] --> B["两阶段优化 (同 H2O)<br/>生成人体与人形姿态对"]
        B --> C["轻量 3 层 MLP<br/>学习人体到机器人姿态的映射"]
    end
    C --> G["Pose-based 模式<br/>全身目标姿态 g_t"]
    V["Root-based 模式<br/>根速度指令 v_t + 上身参考关节角"]
    subgraph DL["II 双层控制器 Dual-level Controller (Dec-POMDP 两个智能体)"]
        X["n 步观测历史<br/>本体状态 + 上一动作 + v_t + g_t"]
        X --> UP["上身策略 π_upper<br/>3 层 MLP"]
        X --> LO["下身策略 π_lower<br/>Gated Transformer-XL (3 层)"]
        UP --> CAT["动作拼接为目标关节位置"]
        LO --> CAT
    end
    G --> X
    V --> X
    CAT --> PD["关节 PD 控制器"]
    subgraph CL["III 课程学习 Curriculum Learning"]
        P1["基础策略 π_base<br/>固定上身关节，只训练下身"] --> P2["监督初始化<br/>上身监督学习模仿参考姿态<br/>下身对 π_base 做 KL 正则 (姿态幅度减半)"]
        P2 --> P3["PPO 强化学习<br/>姿态幅度恢复原值"]
        P3 --> P4["两种模式分别训练后<br/>模仿学习蒸馏为单一网络"]
    end
    P3 -.->|"训练 π_upper 与 π_lower"| DL
```

图依据：论文 §3 / Fig.2（重定向 → 双层控制器 → 课程学习；仿真为 IsaacGym）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/JAEGER__Dual-Level_Humanoid_Whole-Body_Controller` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**JAEGER（Dual-Level Humanoid Whole-Body Controller）在本库作为「双层全身控制器」方向的待读条目存在，双层切分（上身 MLP / 下身 Transformer）见「流程总览」。**

- 标题给出的唯一机制线索是 **dual-level** 结构，分类为 04_Loco-Manipulation_and_WBC，属 WBC 主线。
- 页面无核心机制与实验区块，任何关于其效果的判断都应等深读笔记完成后再下。
- 深读进度追踪指向 **progress.json** 而非 PROGRESS.md，后续 ingest 时注意来源差异。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_jaeger.md](../../sources/papers/humanoid_pnb_jaeger.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2505.06584)：已通读方法部分（实验与附录以略读为准）：§1–§3 方法（重定向网络、双层控制器、课程学习）、Fig.2；§4 实验与附录未读

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
