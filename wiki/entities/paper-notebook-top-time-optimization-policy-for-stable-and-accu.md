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

**TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

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
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2508.00355> |
| 作者 | Zhenghan Chen、Haocheng Xu、Haodong Zhang、Zhongxiang Zhou、Rong Xiong 等（浙江大学） |
| 发表 | 2025 年 8 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**TOP 的核心主张是把「稳 vs 准」的权衡从空间维度挪到时间维度：与其不断加强下身抗扰，不如让上身快速动作在时间上错峰，从源头减少对平衡的冲击。**

- 真正起作用的机制是 **时间轨迹优化 + 解耦控制** 的组合：上身用 PD 换精度、下身用 RL 换鲁棒，VAE 上身动作先验则提供一个紧凑、可被优化的时间轨迹表示。
- 这条思路的隐含前提是 **上身动作的时间安排可以被调整**；若任务对动作时序有外部约束（必须按固定节拍完成），"错峰"这一自由度就不存在，方法收益随之下降。
- 适用边界是 **站立操作**：论文处理的是站立姿态下的平衡-精度-时效三角，本页未涉及行走中操作或移动底座场景。
- 与「一味强化下身抗扰」的路线相比，TOP 明确把后者判为治标不治本；与 Mobile-TeleVision 的相通之处在于同样承认上身与下身对"精确 vs 鲁棒"有不同需求，应分开设计。
- 本页为 **深读笔记编译** 的索引级摘要，"优于已有方法"是定性结论，具体 benchmark、消融与实机指标须以深读笔记与论文 PDF 为准。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_top.md](../../sources/papers/humanoid_pnb_top.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html>
- 论文：<https://arxiv.org/abs/2508.00355>

## 推荐继续阅读

- [机器人论文阅读笔记：TOP](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html)
