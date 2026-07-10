---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2508.00355"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_top.md
summary: "人形能做多样操作，前提是鲁棒精确的站立控制器。已有方法要么难精控高维上身关节、要么难同时保证鲁棒与精度——尤其当上身运动快时。本文提出一个新颖的时间优化策略（Time Optimization Policy, TOP），训练一个站立操作控制模型，同时保证平衡、精度与时间效率。核心思想是：调整上身动作的时间轨迹，而不只是一味强化下身的抗扰能力——让快速上身运动在时间上\"错峰\"，减轻对平衡的冲击。方法用 VAE 编码上身动作先验，并解耦全身控制（上身 PD 控制器 + 下身 RL 控制器）。仿真与真机实验表明，TOP 在站立操作上稳定且精确，优于已有方法。"
---

# TOP

**TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形能做多样操作，前提是鲁棒精确的站立控制器。已有方法要么难精控高维上身关节、要么难同时保证鲁棒与精度——尤其当上身运动快时。本文提出一个新颖的时间优化策略（Time Optimization Policy, TOP），训练一个站立操作控制模型，同时保证平衡、精度与时间效率。核心思想是：调整上身动作的时间轨迹，而不只是一味强化下身的抗扰能力——让快速上身运动在时间上"错峰"，减轻对平衡的冲击。方法用 VAE 编码上身动作先验，并解耦全身控制（上身 PD 控制器 + 下身 RL 控制器）。仿真与真机实验表明，TOP 在站立操作上稳定且精确，优于已有方法。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| TOP | Time Optimization Policy，时间优化策略 |
| Standing Manipulation | 站立操作 |
| VAE | 变分自编码器（编码上身动作先验） |
| Decoupled WBC | 解耦全身控制（上身 PD + 下身 RL） |
| Time Trajectory | 时间轨迹（动作的时间安排） |
| Disturbance Resistance | 抗扰能力 |

## 为什么重要

- **"调时间"是平衡-精度权衡的新维度**：不止调空间动作，还可调时间安排；
- **上身 PD + 下身 RL 解耦**契合"精确 vs 鲁棒"的不同需求，与 Mobile-TeleVision 思路相通；
- **VAE 动作先验**是常用的紧凑表示手段；
- 站立操作是人形干活的基础，稳准快都重要。

## 解决什么问题

站立操作要**平衡 + 精度 + 时间效率**三者兼顾： - 难**精控高维上身关节**； - **上身快速运动**时，扰动大，难同时稳与准； - 一味强化下身抗扰**治标不治本**。

TOP 要：通过**调上身动作时间轨迹**，从源头减轻平衡负担，兼顾稳、准、快。

## 核心机制

1. **时间优化策略 TOP**：调上身动作时间轨迹，同时保证稳/准/快；
2. **VAE 上身动作先验**：紧凑可优化表示；
3. **解耦全身控制**：上身 PD 精控 + 下身 RL 鲁棒；
4. **稳定精确站立操作**：仿真 + 真机优于已有方法。

方法拆解（深读笔记小节）：思想：调上身时间轨迹（而非只强化下身）；VAE 上身动作先验；解耦全身控制；时间优化策略训练；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2508.00355> |
| 作者 | Zhenghan Chen、Haocheng Xu、Haodong Zhang、Zhongxiang Zhou、Rong Xiong 等（浙江大学） |
| 发表 | 2025 年 8 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_top.md](../../sources/papers/humanoid_pnb_top.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html>
- 论文：<https://arxiv.org/abs/2508.00355>

## 推荐继续阅读

- [机器人论文阅读笔记：TOP](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html)
