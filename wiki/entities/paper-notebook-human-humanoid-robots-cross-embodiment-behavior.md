---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2412.15166"
venue: "2024.12"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_human-humanoid-robots-cross-embodiment-behavior.md
summary: "Human-Humanoid Robots Cross-Embodiment Behavior-Skill Transfer Using Decomposed Adversarial Learning from Demonstration：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# Human-Humanoid Robots Cross-Embodiment Behavior-Skill Transfer Using Decomposed Adversarial Learning from Demonstration

**Human-Humanoid Robots Cross-Embodiment Behavior-Skill Transfer Using Decomposed Adversarial Learning from Demonstration** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Human-Humanoid Robots Cross-Embodiment Behavior-Skill Transfer Using Decomposed Adversarial Learning from Demonstration 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 深读笔记尚未撰写；方法流程见下方「流程总览」（依据论文方法部分），具体数字以论文为准。

## 流程总览

```mermaid
flowchart TD
    A["人类示范：Optitrack 动捕 + Manus 手套<br/>物体贴标记点"] --> B["运动学重定向 (Rofunc)<br/>→ 统一数字人 UDH (92 DoF)"]
    B --> C["再重定向到多种人形<br/>按关节功能分组 + 偏 IK + 根轨迹按尺度归一化"]
    C --> D["运动学参考动作"]
    D --> E["DAIL 行为基元预训练<br/>身体分解为下肢 / 上肢 / 手等功能部分<br/>每部分一个行为编码器 + 风格判别器，对抗模仿"]
    E --> F["低层行为控制器 π(a|s, z)<br/>潜行为空间 z (单位超球面)"]
    B --> G["从示范提取交互图 G<br/>节点=身体部位与物体，边=接触二值"]
    G --> H["高层技能策略 η(z|s, G, g) 在潜空间规划行为轨迹<br/>奖励 = 任务 + 交互风格 D_I + 行为风格"]
    F --> H
    H --> I["UDH 上的 loco-manipulation 技能"]
    I --> J["运动学重定向到目标人形"]
    J --> K["MLP 微调层：在行为基元空间生成曲线<br/>运动学参考 → 动力学级控制"]
    F --> K
    K --> L["多种人形机器人上部署同一技能<br/>(NAVIAI / H1 / Walker / Bruce / CURI)"]
```

图依据：论文 §III（A 运动学重定向、B DAIL 行为基元、C 交互图技能学习、D 技能迁移与微调）/ Fig.2、Fig.5

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/Human-Humanoid_Robots_Cross-Embodiment_Behavior-Skill_Transfer_Using_Decomposed_` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前是待读条目而非论文归纳：它只登记了这条「人–人形跨本体技能迁移」条目这一条，实质结论要等读完笔记。**

- 页面现状：status 为 planned，正文只有分类归属与 progress.json 待深读状态，没有深读笔记提炼的指标或实验内容（方法流程见「流程总览」）。
- 从标题能读出的定位仅限于「分解式对抗性示范学习 + 人到人形的行为/技能跨本体迁移」，归入 04_Loco-Manipulation_and_WBC；「分解」如何切分、对抗项怎么设计见「流程总览」。
- 使用方式：把本页当检索入口——分类页 [paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md) 与 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)；要读实质内容请直接走原文。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_human-humanoid-robots-cross-embodiment-behavior.md](../../sources/papers/humanoid_pnb_human-humanoid-robots-cross-embodiment-behavior.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2412.15166)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III 全部小节、§IV-A/B 实验设置与数据采集

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
