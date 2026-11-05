---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2512.23650"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_do-you-have-freestyle-expressive-humanoid-locomo.md
summary: "Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control

**Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」（依据论文方法部分），不给实验结论。

## 流程总览

```mermaid
flowchart LR
    subgraph ALIGN[音频-动作对齐]
        AU[音乐或语音] --> AL[音频 latent l_audio]
        MO[动作] --> ML[预训练 VAE 动作 latent l_motion]
        AL --> AD[音频适配器: 6 层 Transformer + 时间注意力]
        AD -->|InfoNCE 对比学习| ML
    end
    subgraph TEACH[教师策略: 仿真中 RL 训练]
        OBS[机器人本体感知 + 参考动作 + 特权信息] --> GATE[门控网络]
        GATE --> E[ΔMoE 4 个专家, 嵌套条件子空间 S1 到 S4<br/>残差融合 a = w1·a1 + Σ wi·ai-ai-1]
        E --> TA[教师动作 a]
    end
    subgraph STU[学生策略: 扩散模型, DAgger 式蒸馏]
        TXT[文本描述 如 a person is dancing] --> MG[预训练动作生成器 -> 固定 content latent]
        MG --> DIF[扩散去噪 x0 预测]
        AD -->|style: 多层注入 o_i = Layer_i + α·l_audio| DIF
        DIF --> SA[学生动作]
    end
    TA -->|学生 rollout 访问的状态上查询教师动作 监督| DIF
    SA --> G1[Unitree G1 真机部署]
```

图依据：论文 §3.1–3.4（Fig. 2、Fig. 3）、§4 开头。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/do-you-have-freestyle-expressive-humanoid-locomo` |
| arXiv | <https://arxiv.org/abs/2512.23650> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页目前只能支撑一个结论：这是一条已进入 05_Locomotion 深读队列、但尚未产出笔记的条目，它现在的价值只是可检索，而不是内容。**

- 页面能确证的信息仅限分类（05_Locomotion）、arXiv 编号与计划文件夹路径；音频控制如何驱动表现力运动、效果如何，本页均无依据。
- 因此本页 **不可作为技术判断依据**：机制与指标应回到论文原文，或等 PROGRESS.md 追踪的深读笔记完成后再引用。
- 定位是 **待读条目**，方便你按分类找到它；笔记完成后应补成完整摘要，并重写本节为面向内容的结论。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_do-you-have-freestyle-expressive-humanoid-locomo.md](../../sources/papers/humanoid_pnb_do-you-have-freestyle-expressive-humanoid-locomo.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2512.23650>
- [arXiv PDF](https://arxiv.org/pdf/2512.23650)：已通读方法部分（实验与附录以略读为准）：§3.1 概览、§3.2 ΔMoE、§3.3 音频-动作对齐、§3.4 音频条件策略蒸馏

## 推荐继续阅读

- [机器人论文阅读笔记：Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Do_You_Have_Freestyle__Expressive_Humanoid_Locomotion_via_Audio_Control/Do_You_Have_Freestyle__Expressive_Humanoid_Locomotion_via_Audio_Control.html)
- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
