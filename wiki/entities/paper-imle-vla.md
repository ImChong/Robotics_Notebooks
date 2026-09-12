---
type: entity
tags: [paper, vla, imitation-learning, real-time]
status: complete
updated: 2026-09-11
arxiv: "2609.10915"
code: https://kianhk6.github.io/IMLE-VLA/
related:
  - ../methods/vla.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ../overview/dexterous-wm-humanoid-14-papers-technology-map.md
sources:
  - ../../sources/papers/imle-vla_arxiv_2609_10915.md
  - ../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md
summary: "条件 IMLE 单步动作生成器加速 VLA；L40S 55 Hz vs π₀.₅ 15 Hz；LIBERO 四套件均值成功率 98.0%。"
---

# IMLE-VLA（arXiv:2609.10915）

**IMLE-VLA**（[IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies](https://arxiv.org/abs/2609.10915)）来自 [具身智能小站 14 篇盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md)。条件 IMLE 单步动作生成器加速 VLA；L40S 55 Hz vs π₀.₅ 15 Hz；LIBERO 四套件均值成功率 98.0%。

## 一句话定义

**用单步 IMLE 替代多步扩散采样，在保留多模态动作覆盖的同时把 VLA 控制频率拉回实时区间。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| IL | Imitation Learning | 模仿学习 |
| RL | Reinforcement Learning | 强化学习 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

- 纳入本期 **灵巧手 / 世界模型 / 人形控制 / VLA** 主线之一。
- 开源状态：**待发布**（步骤 2.5 核查，2026-09-11）。
- 与 [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.10915](https://arxiv.org/abs/2609.10915) |
| **项目页** | https://kianhk6.github.io/IMLE-VLA/ |
| **代码/资源** | https://kianhk6.github.io/IMLE-VLA/ |
| **开源** | **待发布** |
| **文内指标** | LIBERO 40 任务×50 次均值 98.0%；L40S 55 Hz；H=30 时动作吞吐 11.0×（更长开环牺牲反应性）。 |


## 源码运行时序图

**不适用**（项目页列 Code 入口；截至入库日未在页上找到独立 GitHub 仓库链接。）

## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | LIBERO 40 任务×50 次均值 98.0%；L40S 55 Hz；H=30 时动作吞吐 11.0×（更长开环牺牲反应性）。 |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准（[参考来源](#参考来源)）。

## 与其他工作对比

- **多步去噪的 [Diffusion Policy](../methods/diffusion-policy.md) / π₀ 家族** — 采样需多步迭代，控制频率受限（文内口径 π₀.₅ 约 15 Hz）；IMLE-VLA 用 **条件 IMLE 单步生成** 换实时性（L40S 约 55 Hz），同时保留多模态动作覆盖。
- **[Action Chunking](../methods/action-chunking.md)** — 靠 **加长开环视界** 摊薄推理开销；IMLE-VLA 文内指出 H=30 虽有 11.0× 动作吞吐，但更长开环牺牲反应性——两条加速路径的代价不同。
- **[DLSRL](./paper-dlsrl.md)** — 同样动生成式策略的表示层，但目标是 **在线 RL 适配**（冻结生成器 + 双 latent）；IMLE-VLA 换的是 **采样器本身**。
- **[Show-Harness](./paper-show-harness.md)** — 通过离散语义微动作接口绕开策略推理开销；IMLE-VLA 仍训端到端策略，只压缩采样步数。
- **[模仿学习 vs 强化学习](../comparisons/rl-vs-il.md)** — 本文落在 IL 侧的生成式策略分支，评测以 LIBERO 四套件均值成功率（98.0%）为口径。

- **读法：** 以上为知识库内 **路线级** 对照；与原文 baseline 的逐项定量比较与消融以 **原文 PDF** 为准（[参考来源](#参考来源)）。

## 结论

**IMLE-VLA 适合作为本期「待发布」边界下的快速索引页，部署前请核对仓库/README 可运行性。**

1. 核心贡献：用单步 IMLE 替代多步扩散采样，在保留多模态动作覆盖的同时把 VLA 控制频率拉回实时区间。
2. 开源结论：**待发布** — 以项目页实际链接为准（入库日 2026-09-11）。
3. 横向对照见 [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [imle-vla_arxiv_2609_10915.md](../../sources/papers/imle-vla_arxiv_2609_10915.md)
- [wechat 14篇盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md)
- [arXiv:2609.10915](https://arxiv.org/abs/2609.10915)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.10915)
- [项目页/资源](https://kianhk6.github.io/IMLE-VLA/)
