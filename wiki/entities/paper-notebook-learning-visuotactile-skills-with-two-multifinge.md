---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2404.16823"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-visuotactile-skills-with-two-multifinge.md
summary: "为复刻人类的灵巧、感知体验与动作模式，本文用一套带多指手与视触觉数据的双手系统，从人类演示学习。为解决采集训练数据的硬件难题，作者开发了低成本遥操作系统 HATO（用现成部件搭建，高效采集双手数据），并把义肢手（prosthetic hands）改装、加装触觉传感器。在需多指灵巧的长时程、高精度操作任务上做模仿学习，并通过消融研究考察数据规模、感知模态（视觉/触觉）重要性、视觉预处理的影响。结果证明：结合视觉与触觉反馈能让机器人从人类演示学到复杂双手操作技能，推进多指灵巧控制的可行性。"
---

# Learning Visuotactile Skills with Two Multifingered Hands

**Learning Visuotactile Skills with Two Multifingered Hands** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

为复刻人类的灵巧、感知体验与动作模式，本文用一套带多指手与视触觉数据的双手系统，从人类演示学习。为解决采集训练数据的硬件难题，作者开发了低成本遥操作系统 HATO（用现成部件搭建，高效采集双手数据），并把义肢手（prosthetic hands）改装、加装触觉传感器。在需多指灵巧的长时程、高精度操作任务上做模仿学习，并通过消融研究考察数据规模、感知模态（视觉/触觉）重要性、视觉预处理的影响。结果证明：结合视觉与触觉反馈能让机器人从人类演示学到复杂双手操作技能，推进多指灵巧控制的可行性。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Visuotactile | 视触觉（视觉 + 触觉） |
| Multifingered | 多指（灵巧手） |
| HATO | 本文低成本双手遥操作系统 |
| Prosthetic Hand | 义肢手（改装 + 触觉传感器） |
| Imitation Learning | 模仿学习 |
| Ablation | 消融研究 |

## 为什么重要

- **触觉对多指灵巧操作的贡献被消融量化**，为"该不该上触觉"提供证据；
- **低成本硬件（义肢手 + 现成件）**降低双手灵巧研究门槛；
- 对人形双手操作直接相关；
- 与"人形视触觉数据集"等触觉工作共同强调触觉模态。

## 解决什么问题

双手多指**视触觉**操作难采数据、难学： - 多指手 + 触觉传感**硬件贵、难搭**； - 缺**低成本**采集系统； - 不清楚**触觉/视觉/数据规模**各自的贡献。

论文要：低成本双手视触觉系统 + 从人类演示学复杂操作，并厘清各模态贡献。

## 核心机制

1. **HATO 低成本双手遥操作系统**：现成部件 + 触觉义肢手；
2. **视触觉模仿学习**：长时程高精度多指任务；
3. **系统消融**：数据规模、视觉/触觉模态、视觉预处理；
4. **视觉+触觉协同**：学到复杂双手操作技能。

方法拆解（深读笔记小节）：HATO 低成本双手遥操作 + 触觉义肢手；视触觉模仿学习；消融研究；结论；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_Visuotactile_Skills_with_Two_Multifingered_Hands/Learning_Visuotactile_Skills_with_Two_Multifingered_Hands.html> |
| arXiv | <https://arxiv.org/abs/2404.16823> |
| 作者 | Toru Lin、Yu Zhang、Qiyang Li、Haozhi Qi、Brent Yi、Sergey Levine、Jitendra Malik（UC Berkeley） |
| 发表 | 2024 年 4 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-visuotactile-skills-with-two-multifinge.md](../../sources/papers/humanoid_pnb_learning-visuotactile-skills-with-two-multifinge.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_Visuotactile_Skills_with_Two_Multifingered_Hands/Learning_Visuotactile_Skills_with_Two_Multifingered_Hands.html>
- 论文：<https://arxiv.org/abs/2404.16823>

## 推荐继续阅读

- [机器人论文阅读笔记：Learning Visuotactile Skills with Two Multifingered Hands](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_Visuotactile_Skills_with_Two_Multifingered_Hands/Learning_Visuotactile_Skills_with_Two_Multifingered_Hands.html)
