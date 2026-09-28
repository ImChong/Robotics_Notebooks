---
type: entity
tags: [paper, humanoid-paper-notebooks, manipulation, humanoid, sim2real, co-training, synthetic-data, diffusion-policy, nvidia, ut-austin, berkeley, fourier]
status: complete
updated: 2026-09-28
arxiv: "2503.24361"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../concepts/sim2real.md
  - ./paper-diffusion-policy.md
  - ./paper-notebook-robocasa-large-scale-simulation-of-everyday-task.md
  - ./paper-notebook-dexmimicgen-automated-data-generation-for-bimanu.md
  - ./paper-notebook-a-systematic-study-of-data-modalities-and-strate.md
sources:
  - ../../sources/papers/humanoid_pnb_sim-and-real-co-training.md
summary: "大规模真实机器人数据集潜力大，但真实人类数据采集费时费力。本文主张：与其只做 sim-to-real 迁移，不如在训练时直接把「仿真」与「真实」数据集混合协同训练（co-training）。通过在机械臂与人形系统、多样操作任务上的系统实验，作者证明：即便仿真与真实数据有明显差异，仿真数据也能把真实任务表现平均提升 38%。这给出一个简单有效的视觉操作训练配方。"
---

# Sim-and-Real Co-Training

**Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

大规模真实机器人数据集潜力大，但真实人类数据采集费时费力。本文主张：与其只做 sim-to-real 迁移，不如在训练时直接把「仿真」与「真实」数据集混合协同训练（co-training）。通过在机械臂与人形系统、多样操作任务上的系统实验，作者证明：即便仿真与真实数据有明显差异，仿真数据也能把真实任务表现平均提升 38%。这给出一个简单有效的视觉操作训练配方。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Co-Training | 协同训练，sim 与 real 数据混合训练 |
| Sim-to-Real | 仿真到真机迁移（被对比对象） |
| Vision-Based | 基于视觉的策略 |
| Domain Gap | 域差距（sim 与 real 差异） |
| Recipe | 配方，可复用的训练做法 |
| Generalist | 通才机器人模型 |

## 为什么重要

- **"混合训练 > 两段迁移"是反直觉但实用的洞见**：让模型同时见两域更稳；
- **仿真数据即便不完美也有用**，降低对昂贵真实数据的依赖；
- 对人形（真实采集更难）尤其有价值；
- 与 DreamGen、DexMimicGen 等"用仿真/合成数据扩规模"思路互补。

## 解决什么问题

真实数据采集贵，仿真数据多但有域差距： - 纯 **sim-to-real 迁移**常需精心对齐； - 想更**简单**地用上仿真数据提升真实表现。

论文要：一个**简单配方**——直接 **sim + real 协同训练**，看仿真数据能否稳定帮真实任务。

## 核心机制

1. **sim+real 协同训练配方**：训练时混合，而非两段迁移；
2. **系统实验（臂 + 人形）**：研究最优配方；
3. **+38% 真实表现**：即便域差异明显；
4. **简单可复用**：易嫁接到现有视觉操作流程。

方法拆解（深读笔记小节）：协同训练（sim + real 混合）；系统实验（臂 + 人形）；结论；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2503.24361> |
| 源码 | **未开源**：项目页 <https://co-training.github.io/> 的 Code 按钮被注释掉（原指向 DexMimicGen 模板链接），截至 2026-09-28 未见本文代码 |
| 作者 | Abhiram Maddukuri、Zhenyu Jiang、Soroush Nasiriany、Ken Goldberg、Ajay Mandlekar、Linxi Fan、Yuke Zhu 等（NVIDIA / Berkeley / UT） |
| 发表 | 2025 年 3 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：两个真实域——Panda Kitchen（Franka + DROID 桌面硬件，3 个任务各 50 条演示）与 Humanoid Tabletop（Fourier GR-1，mink IK 控制、头部第一视角相机，3 个任务各 20 条演示）。仿真数据两类：任务无关的先验仿真数据（Prior）与为每个真实任务粗略搭建的「数字表亲」数据（DC，任务定义与相机视角近似对齐）。策略为 Diffusion Policy（RGB + 本体）。

| 数据组合（Table I） | C2SPnP | C2CPnP | CloseDoor | CupPnP | MilkPnP | Pouring | 平均 |
|------|---:|---:|---:|---:|---:|---:|---:|
| 仅真实 | 44% | 38% | 10% | 65% | 50% | 65% | 45.3% |
| 真实 + DC | 67% | 72% | 100% | 95% | 70% | 85% | 81.1% |
| 真实 + Prior | 58% | 53% | 100% | 80% | 80% | 70% | 76.8% |
| **真实 + DC + Prior** | 72% | 72% | 100% | 85% | 80% | 90% | **83.2%** |

（前三列为 Panda，后三列为 GR-1。）

- **泛化**（Table II）：未见物体 Panda 33% → 50%、GR-1 10% → 80%；未见位置 Panda 11% → 28%、GR-1 43% → 100%（真实数据中刻意去掉了中心位置的演示）。
- **数据富足时仍有效**：人形 MultiTaskPnP 固定 4000 条 DC，真实演示从 40 加到 400，协同训练始终优于只用真实数据。
- **仿真数据量**：DC 从 1 万条降到 500 条，Panda 67% → 53%；GR-1 从 1000 条降到 100 条，95% → 75%。
- **协同比例**（每个 minibatch 采样仿真的概率）：1:1 次优，**99%** 最好；再提到 99.5% / 99.9% 则从 95% 掉到 60%。
- **相机对齐**：用默认未对齐视角渲染 DC，Panda 67% → 56%、GR-1 95% → 70%；但不需要完全对齐（真机鱼眼畸变未建模也可）。

## 与其他工作对比

| 工作 | 仿真数据的用法 | 与本文的差异 |
|------|------|------|
| 先仿真训练再迁移（传统 Sim2Real） | 仿真预训练 → 真机微调 / 域随机化 | 本文在同一训练中混合两域，不追求精确对齐 |
| [DexMimicGen](./paper-notebook-dexmimicgen-automated-data-generation-for-bimanu.md) | 从少量演示在仿真中自动扩增 | 解决「仿真数据从哪来」；本文解决「怎么和真实数据混」 |
| [DreamGen](./paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md) | 视频世界模型生成合成轨迹 | 适合难仿真的可变形 / 液体任务，正是本文自述的局限 |
| [A Systematic Study of Data Modalities](./paper-notebook-a-systematic-study-of-data-modalities-and-strate.md) | 大行为模型协同训练的多模态数据研究 | 更大规模、更多模态；本文聚焦单任务 sim + real 配方 |

## 结论

**这篇的价值在于「少做一件事」：把 sim-to-real 的两段迁移换成训练时直接混合 sim 与 real 数据，用配方的简单性换掉精心域对齐的工程成本。**

- 真正的结论性指标是**真实任务表现平均 +38%**（45.3% → 83.2%），而且即便用与真实任务无关的先验仿真数据也有 +31.5%——这正是「必须先把域差距对齐」这一直觉的反例。
- 起作用的机制是让模型在训练中同时见到两域，而不是先在仿真里学好再迁；但「简单」不等于免调参：协同比例 99% 最好，1:1 次优，过高又会大幅下降。
- 适用范围由实验覆盖界定：机械臂与人形系统、多样操作任务、基于视觉的策略；本页没有承诺在此之外同样成立，最优混合配方也需按系统重新试。
- 对人形尤其划算：真实采集本就更难，这条配方直接降低对昂贵真机数据的依赖。
- 与 DreamGen、DexMimicGen 等「用仿真/合成数据扩规模」的路线互补——那些工作解决数据从哪来，本文解决数据怎么混。

## 局限与风险

- **任务类型偏窄**：多为取放，高精度插装与更长时程任务未验证（论文自述）。
- **成功率仍不完美**：最好的配方平均 83.2%。
- **难仿真的任务受限**：可变形物体、液体等难以准确仿真，作者建议改用视频生成 / 世界模型数据。
- **超参敏感**：协同比例需逐任务调，99% 与 99.9% 之间可差 35 个百分点；仿真数据量需比真实数据高出数量级。
- **开源边界**：未见代码；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- Sim2Real 概念：[sim2real](../concepts/sim2real.md)
- 策略骨架 Diffusion Policy：[paper-diffusion-policy](./paper-diffusion-policy.md)
- 任务无关先验仿真数据来源 RoboCasa：[paper-notebook-robocasa-large-scale-simulation-of-everyday-task](./paper-notebook-robocasa-large-scale-simulation-of-everyday-task.md)
- 仿真演示自动扩增：[paper-notebook-dexmimicgen-automated-data-generation-for-bimanu](./paper-notebook-dexmimicgen-automated-data-generation-for-bimanu.md)
- LBM 协同训练数据模态系统研究：[paper-notebook-a-systematic-study-of-data-modalities-and-strate](./paper-notebook-a-systematic-study-of-data-modalities-and-strate.md)

## 参考来源

- [humanoid_pnb_sim-and-real-co-training.md](../../sources/papers/humanoid_pnb_sim-and-real-co-training.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation.html>
- 论文：<https://arxiv.org/abs/2503.24361>
- 论文正文（Table I–II、关键要素与配方节）：<https://arxiv.org/html/2503.24361>

## 推荐继续阅读

- [机器人论文阅读笔记：Sim-and-Real Co-Training](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation/Sim-and-Real_Co-Training__A_Simple_Recipe_for_Vision-Based_Robotic_Manipulation.html)
- 项目页：<https://co-training.github.io/>
