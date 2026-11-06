---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2509.13534"
venue: "2025.09"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_embracing-bulky-objects-with-humanoid-robots.md
summary: "Embracing Bulky Objects with Humanoid Robots：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# Embracing Bulky Objects with Humanoid Robots

**Embracing Bulky Objects with Humanoid Robots: Whole-Body Manipulation with Reinforcement Learning** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Embracing Bulky Objects with Humanoid Robots 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    A["AMASS 动捕 (SMPL)<br/>MaskedMimic 过滤得到可行人体运动 D_h"] --> B["H2O 重定向到人形机器人<br/>得到机器人运动数据集 D_r"]
    B --> C["教师策略 π_teacher（PHC 框架，PPO，MLP）<br/>输入本体状态 + 参考运动目标 → 精确跟踪动作 a*"]
    C --> D["VAE 式运动先验蒸馏（PULSE）<br/>编码器 E / 解码器 D / 可学习先验 R<br/>L = L_action + α·L_regu + β·L_KL"]
    D --> E["冻结解码器 D 与先验 R<br/>得到低维潜在动作空间"]
    F["任务观测 s_task = 解码器输入 s_d + 任务特征 s_s<br/>箱体/目标距离与方向、NSDF 特征 d_t"] --> G["任务策略 π_task（PPO）<br/>输出潜在动作 z_t"]
    G --> H["z_t + 先验均值 μ_p → 解码器 D → 机器人动作 a_t"]
    E --> H
    H --> I["仿真中接近 → 抱起 → 搬运箱体"]
    I --> J["奖励：任务 (walk/carry/arm/NSDF) + 平滑 + 物理限制"]
    J --> G
    K["随机初始化三类场景：<br/>远处接近 / 拾取区预抱姿态 / 已抱住物体"] --> I
```

图依据：论文 §III-A–C / Fig.2、Fig.3，式 (1)–(11)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/Embracing_Bulky_Objects_with_Humanoid_Robots__Whole-Body_Manipulation_with_Reinf` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Embracing Bulky Objects with Humanoid Robots 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2509.13534（2025.09）、分类 04_Loco-Manipulation_and_WBC、计划中的笔记文件夹；论文的方法机制、评测口径与实机结果在本页一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_embracing-bulky-objects-with-humanoid-robots.md](../../sources/papers/humanoid_pnb_embracing-bulky-objects-with-humanoid-robots.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2509.13534)：已通读方法部分（实验与附录以略读为准）：§III 方法（III-A–C）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
