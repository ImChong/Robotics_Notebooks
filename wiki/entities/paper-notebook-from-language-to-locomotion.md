---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2510.14952"
venue: "2025.10"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_from-language-to-locomotion.md
summary: "From Language to Locomotion：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# From Language to Locomotion

**From Language to Locomotion: Retargeting-free Humanoid Control via Motion Latent Guidance** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

From Language to Locomotion 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」，不含实验结论。

## 流程总览

```mermaid
flowchart TD
    T["文本指令 T（如 A person walks in a circle）"] --> G["阶段一：连续自回归运动生成器（LaMP 文本编码 + Causal Transformer 掩码预测 + Diffusion 头）"]
    G --> L["运动潜变量 l_ref（不解码为显式运动序列，无需重定向）"]
    D["筛选后的重定向动作数据集（初始策略 π0 跟踪误差 e(s)>0.6 的序列被剔除）"] --> TP["阶段二：MoE 教师策略 π_t（PPO，特权信息 + 参考动作，输出 23 维目标关节位置经 PD 控制）"]
    CAS["Causal Adaptive Sampling：按失败区间向前归因，提高前因区间采样概率"] --> TP
    TP -->|"DAgger 式 rollout，查询教师动作"| ST["扩散学生策略 π_s：对教师动作加噪，以 l_ref + 本体感知 + 历史观测经 latent encoder 条件去噪（x0 预测，MSE）"]
    L --> ST
    ST --> INF["部署推理：高斯噪声 + DDIM 采样 + AdaLN 以 l_ref、本体感知、历史观测为条件"]
    INF --> ACT["动作 a 经 PD 控制直接驱动人形机器人"]
    L -.->|"推理时仅需 l_ref"| INF
```

图依据：论文 §3.1 Overview、§3.2–3.4 / Fig.2（RoboGhost）；PD 控制输出见 §3.3.1

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/From_Language_to_Locomotion__Retargeting-free_Humanoid_Control_via_Motion_Latent` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**From Language to Locomotion 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2510.14952（2025.10）、分类 04_Loco-Manipulation_and_WBC、计划中的笔记文件夹；论文的方法机制、评测口径与实机结果在本页一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_from-language-to-locomotion.md](../../sources/papers/humanoid_pnb_from-language-to-locomotion.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2510.14952)：已通读方法部分（实验与附录以略读为准）：§3 Method（3.1–3.4）与 Fig.2


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
