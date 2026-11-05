---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2509.18046"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humam-humanoid-motion-control-via-end-to-end-dee.md
summary: "HuMam：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# HuMam

**HuMam: Humanoid Motion Control via End-to-End Deep RL with Mamba** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

HuMam 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    A["机器人自身状态 s_robot<br/>腿部 12 关节位置/速度、根部 roll/pitch、角速度"] --> C["拼接状态 s_t"]
    B["外部状态 s_ext<br/>后续 2 个规划落脚点 + 步态时钟 (sin, cos)"] --> C
    C --> D["投影模块 f_proj<br/>得到 robot / ext 两个 token 嵌入"]
    D --> E["单层 Mamba 编码器<br/>门控状态空间动态，仅用当前状态"]
    E --> F["聚合特征 h_t = f_head([y_robot; y_ext])"]
    F --> G["策略头 π：期望关节位置 a_t ∈ R12<br/>（40Hz，叠加名义半坐姿偏置）"]
    F --> V["价值头 V"]
    G --> H["低增益 PD 控制器 (1000Hz)<br/>转为关节力矩"]
    H --> I["仿真环境 / 人形机器人"]
    I --> J["奖励 R_t：足端接触力、摆动速度、落脚精度<br/>+ 朝向、身高、上身稳定（加权和）"]
    I --> C
    J --> K["PPO 更新（共享 Mamba 表示的策略与价值）"]
    V --> K
    K --> E
```

图依据：论文 §3.1–3.4 / Fig.1，式 (2)–(5)、(12)–(17)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/humam-humanoid-motion-control-via-end-to-end-dee` |
| arXiv | <https://arxiv.org/abs/2509.18046> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 HuMam 的阅读进度待读，不是方法归纳：页面里可确定的只有它被归入 05_Locomotion 与论文入口，端到端 RL 的具体设计与收益都待深读笔记补齐。**

- 页面正文未给出任何方法、指标或消融，「实验与评测」已明确标注待笔记完成后补充。
- 适用边界：可用于按分类检索与追踪阅读进度，不可当作 HuMam 的结论来引用。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humam-humanoid-motion-control-via-end-to-end-dee.md](../../sources/papers/humanoid_pnb_humam-humanoid-motion-control-via-end-to-end-dee.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2509.18046>
- [arXiv PDF](https://arxiv.org/pdf/2509.18046)：已通读方法部分（实验与附录以略读为准）：§3 Methodology（3.1–3.4）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
