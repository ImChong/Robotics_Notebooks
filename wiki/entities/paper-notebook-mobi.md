---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2505.23692"
venue: "2025.05"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_mobi.md
summary: "Mobi-π：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# Mobi-π

**Mobi-π: Mobilizing Your Robot Learning Policy** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Mobi-π 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见「流程总览」（依据论文方法部分），不给实验结论。

## 流程总览

```mermaid
flowchart TD
    SCAN["进入新场景：约 1000 张 RGB-D 图像"]
    GS["训练 3D Gaussian Splatting 场景模型"]
    POSE["候选机器人底座位姿 p → 相机位姿 v"]
    RENDER["3DGS 渲染该视角图像"]
    KID["K_id：DINO 特征与演示初始观测的 KNN 距离（是否分布内）"]
    KOBJ["K_obj：MiniCPM-v2 判断目标物体是否可见"]
    KCOL["K_col：占据地图判断位姿是否无碰撞"]
    SCORE["混合得分 K(p)：K_obj 与 K_col 均为 1 时取 K_id，否则为 0"]
    BO["贝叶斯优化 BO：提出新候选位姿并更新代理模型"]
    BEST["得分最高的底座位姿 p_π"]
    RUN["导航到 p_π 并执行预训练操作策略"]
    SCAN --> GS
    BO --> POSE --> RENDER
    GS --> RENDER
    RENDER --> KID
    RENDER --> KOBJ
    POSE --> KCOL
    KID --> SCORE
    KOBJ --> SCORE
    KCOL --> SCORE
    SCORE -->|"反馈得分"| BO
    BO -->|"迭代结束"| BEST --> RUN
```

图依据：论文 §4（Fig.2、式 1–4 与 BO 段落）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/Mobi-π__Mobilizing_Your_Robot_Learning_Policy` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是尚未深读的待读条目：Mobi-π 在清单里已有一条，但本页还不提供任何可引用的技术结论。**

- 该条目已列入 Paper Notebooks 的 progress 待深读清单，分类为 04_Loco-Manipulation_and_WBC，可从总索引与分类页检索到。
- 深读笔记尚未撰写，本页除「流程总览」外不含机制拆解，也不含量化 benchmark 或实机指标；需要判断这篇工作请直接回到 arXiv 原文。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_mobi.md](../../sources/papers/humanoid_pnb_mobi.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2505.23692)：已通读方法部分（实验与附录以略读为准）：§3（问题定义）、§4（方法）、§5.1 / §5.2 与 §6 的基线设定（实验数字略读）


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
