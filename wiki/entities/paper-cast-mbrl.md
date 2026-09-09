---
type: entity
tags: ['paper', 'mbrl', 'reinforcement-learning', 'locomotion']
status: complete
updated: 2026-09-09
arxiv: "2609.08853"
venue: "arXiv 2026"
related:
  - ../overview/visual-focus-data-efficiency-10-papers-technology-map.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/cast_mbrl_arxiv_2609_08853.md
  - ../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md
summary: "CAST（arXiv:2609.08853）：交替 state-value 目标把规划器行为与策略想象轨迹接起来；DMControl+HumanoidBench 1M 步均值 764±63；Go2 真机倒立迁移。"
---

# CAST

**CAST**（*Alternating State-Value Targets and Expanded Policy Gradients for Model-Based Reinforcement Learning*，[arXiv:2609.08853](https://arxiv.org/abs/2609.08853)，[项目/代码](https://pietronoah.github.io/cast/)）— 详见 [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)。

## 一句话定义

在线规划器通常比当前策略强，但价值函数若只学策略本身就浪费了规划器经验——CAST 用交替 Bellman 目标把两者绑成唯一不动点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAST | Critic with Alternating State-value Target | 本文 MBRL 价值学习 |
| MBRL | Model-Based Reinforcement Learning | 模型式强化学习 |
| MPPI | Model Predictive Path Integral | TD-MPC2 族规划器 |
| SR | Success Rate | 任务回报/成功率 |

## 为什么重要

- 14 任务 1M 步：均值回报 764±63，高于 BMPC 644±43、BOOM 721±41
- Q→V critic + 扩展 k 步策略梯度

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.08853](https://arxiv.org/abs/2609.08853) |
| **开源** | **未开源** |
| **项目/代码** | [https://pietronoah.github.io/cast/](https://pietronoah.github.io/cast/) |

## 核心原理

- 14 任务 1M 步：均值回报 764±63，高于 BMPC 644±43、BOOM 721±41
- Q→V critic + 扩展 k 步策略梯度
- 仿真训练策略零样本迁移 Unitree Go2 动态倒立

## 源码运行时序图

**不适用（官方可运行代码尚未发布或待核实）。** 截至 2026-09-09 以项目页/公众号链为准。

## 实验与评测

- 指标与设置以原文 PDF / 项目页为准；上文 Highlights 来自公众号归纳 + 项目页摘要。
- 横向对照见 [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)。

## 结论

**CAST 的可迁移主张已写入 Highlights；部署前以原文实验设定与开源边界为准。**

1. **真影响：** 见核心原理 bullets。
2. **次要代价：** 预印本/待开源项需独立复现验证。
3. **部署读法：** 未开源 — 先读 README 或项目页再接真机/智能体栈。

## 关联页面

- [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)
- [模仿学习](../methods/imitation-learning.md)

## 参考来源

- [cast_mbrl_arxiv_2609_08853.md](../../sources/papers/cast_mbrl_arxiv_2609_08853.md)
- [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- [arXiv:2609.08853](https://arxiv.org/abs/2609.08853)

## 推荐继续阅读

- [原文 PDF](https://arxiv.org/pdf/2609.08853)
- [项目/代码](https://pietronoah.github.io/cast/)
