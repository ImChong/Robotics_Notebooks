---
type: entity
tags:
  - paper
  - world-models
  - data-collection
  - controllability
  - manipulation
status: complete
updated: 2026-10-22
project_id: aim-action-influence-maximization
arxiv: "2610.27387"
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../overview/hub-embodied-foundation-model.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md
  - ../../sources/papers/aim_action_influence_maximization_arxiv_2610_27387.md
summary: "AIM（2026）：指出动作条件视频世界模型忽略动作的根因在训练数据的动作影响不足，并主动采集数据以最大化动作影响，提高末端执行器可控性的数据效率。"
---

# Action Influence Maximization（AIM，提升世界模型动作可控性的数据采集）

## 一句话定义

**AIM 认为动作条件世界模型「不听话」的根因在数据：专家示范里动作影响太小，因此主动采集能产生更多动作影响的 episode，而不是只扩大模型。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AIM | Action Influence Maximization | 本文提出的主动数据采集目标 |
| WM | World Model | 动作条件视频预测模型 |
| EE | End-Effector | 末端执行器，本文的可控性评测对象 |

## 为什么重要

- 世界模型常预测「典型的专家式未来」，而不是随动作变化的未来，这会削弱其在规划与策略评估中的作用。
- 作者证明数据的动作影响限制了任何在其上训练的模型所能表达的动作影响，因此单靠模型容量无效。

## 方法与机制

- 主动采集训练 episode，目标是提高数据集的动作影响。
- 与探索和内在动机基线比较，AIM 的目标直接针对可控性，而非泛化的新颖性。

## 实验与评测

- 设置：像素级操作任务。
- 指标：末端执行器可控性，以数据效率衡量（据摘要）。

## 与其他工作对比

- 与探索（exploration）与内在动机（intrinsic motivation）基线相比，AIM 在数据效率上更优（据摘要）。
- 与「增大模型容量」路线相对：作者的理论论证支持数据侧改动。

## 结论

本文把动作可控性问题从模型侧转到数据侧。

- 若世界模型忽略动作，先检查训练数据的动作影响，而不是先加大模型。
- 主动采集应以动作影响为目标，而不是只追求状态覆盖。
- 证据目前来自摘要与像素级操作任务；尚未核实在其他观测形式或本体上的表现。

## 工程实践

- 代码与项目页：抓取页面未列出。
- 工程上需要能在线评估动作影响的指标（推测，摘要未展开具体计算方式）。

## 局限与风险

- 摘要未给出具体的动作影响度量公式，复现需要读全文。
- 推测：主动采集可能偏向模型当前不确定的区域，需要防止数据分布过窄。

## 关联页面

- [World Action Models](../concepts/world-action-models.md) — 动作条件世界模型的背景
- [Generative World Models](../methods/generative-world-models.md) — 视频世界模型的方法族

## 参考来源

- [AIM 论文来源归档](../../sources/papers/aim_action_influence_maximization_arxiv_2610_27387.md)

## 推荐继续阅读

- [arXiv 2610.27387](https://arxiv.org/abs/2610.27387)
