---
type: entity
tags: ['paper', 'hri', 'dialogue', 'bayesian']
status: complete
updated: 2026-09-09
arxiv: "2609.08678"
venue: "arXiv 2026"
related:
  - ../overview/visual-focus-data-efficiency-10-papers-technology-map.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/hibridge_dialogue_arxiv_2609_08678.md
  - ../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md
summary: "HiBRIDGE（arXiv:2609.08678）：层级贝叶斯神经网络表达群体对话中「对谁说话/说什么」的不确定性；离线+在线+实时群体交互。"
---

# HiBRIDGE

**HiBRIDGE**（*A Hierarchical Bayesian Neural Network Framework for Interpretable Dialogue Management in Group-Robot Interaction*，[arXiv:2609.08678](https://arxiv.org/abs/2609.08678)，[项目/代码](https://github.com/Massimilianonigro/HiBridge)）— 详见 [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)。

## 一句话定义

多人对话里多个行为都合理时，确定性分类器无法表达犹豫——HiBRIDGE 用层级贝叶斯 + 决策树代理给出可解释的不确定性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HiBRIDGE | Hierarchical Bayesian dialogue management | 本文框架 |
| HRI | Human-Robot Interaction | 群体机器人交互 |
| BNN | Bayesian Neural Network | 概率预测骨干 |
| DM | Dialogue Management | 对话决策模块 |

## 为什么重要

- 覆盖离线标注、在线评价、实时群体交互三阶段
- 决策树代理解释贝叶斯预测

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.08678](https://arxiv.org/abs/2609.08678) |
| **开源** | **待核实** |
| **项目/代码** | [https://github.com/Massimilianonigro/HiBridge](https://github.com/Massimilianonigro/HiBridge) |

## 核心原理

- 覆盖离线标注、在线评价、实时群体交互三阶段
- 决策树代理解释贝叶斯预测
- 公众号列 GitHub 链 404（2026-09-09 复核），按待核实处理

## 源码运行时序图

**不适用（官方可运行代码尚未发布或待核实）。** 截至 2026-09-09 以项目页/公众号链为准。

## 实验与评测

- 指标与设置以原文 PDF / 项目页为准；上文 Highlights 来自公众号归纳 + 项目页摘要。
- 横向对照见 [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)。

## 结论

**HiBRIDGE 的可迁移主张已写入 Highlights；部署前以原文实验设定与开源边界为准。**

1. **真影响：** 见核心原理 bullets。
2. **次要代价：** 预印本/待开源项需独立复现验证。
3. **部署读法：** 待核实 — 先读 README 或项目页再接真机/智能体栈。

## 关联页面

- [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)
- [模仿学习](../methods/imitation-learning.md)

## 参考来源

- [hibridge_dialogue_arxiv_2609_08678.md](../../sources/papers/hibridge_dialogue_arxiv_2609_08678.md)
- [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- [arXiv:2609.08678](https://arxiv.org/abs/2609.08678)

## 推荐继续阅读

- [原文 PDF](https://arxiv.org/pdf/2609.08678)
- [项目/代码](https://github.com/Massimilianonigro/HiBridge)
