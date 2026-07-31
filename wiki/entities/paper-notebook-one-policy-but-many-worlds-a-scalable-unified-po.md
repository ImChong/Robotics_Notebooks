---
type: entity
tags: [paper, humanoid, world-model, diffusion, locomotion, hmi-papers, humanoid-paper-notebooks]
status: complete
updated: 2026-07-31
arxiv: "2505.18780"
venue: "HMI curated · Paper Notebooks"
related:
  - ../concepts/world-action-models.md
  - ../methods/dreamwaq.md
  - ../tasks/humanoid-locomotion.md
  - ./paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md
  - ../queries/hmi-papers-coverage.md
  - ../overview/paper-notebook-category-05-locomotion.md
sources:
  - ../../sources/papers/hmi_p018_dreampolicy-humanoid-locomotion.md
  - ../../sources/papers/humanoid_pnb_one-policy-but-many-worlds-a-scalable-unified-po.md
  - ../../sources/repos/humanoid-motion-intelligence.md
summary: "DreamPolicy / One Policy but Many Worlds（arXiv:2505.18780，HMI P018）：自回归扩散世界模型生成未来状态，goal-conditioned RL 学统一多地形人形行走策略。"
---

# DreamPolicy（One Policy but Many Worlds，HMI P018）

**One Policy but Many Worlds / DreamPolicy**（[arXiv:2505.18780](https://arxiv.org/abs/2505.18780)，[项目页](https://dreampolicy.github.io/)）收录于 HMI **P018**。本页为该 arXiv 的**唯一详情节点**（同时承接 Paper Notebooks 占位名）。

## 一句话定义

先采多地形专家数据训自回归扩散世界模型生成未来状态，再以目标条件 RL 学统一跟踪策略，减少混合地形重复奖励工程。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 自回归扩散生成未来身体状态 |
| AMP | Adversarial Motion Prior | 在线判别器约束长时风格 |
| RL | Reinforcement Learning | goal-conditioned 统一策略 |
| PD | Proportional–Derivative | 关节目标跟踪 |

## 为什么重要

- 针对「每种地形一个专家」的扩展瓶颈，把统一阶段从重复奖励工程里解放出来。
- 生成的是未来状态而非最终力矩，保留低层纠偏空间。
- HMI 解读与 Paper Notebooks 占位合并，避免双节点。

## 核心原理

title: "DreamPolicy: A Unified World-model Policy for Scalable Humanoid Locomotion" track: "世界模型VLA与Agent"

```mermaid
flowchart LR
  A["多地形专家数据"] --> B["自回归扩散 WM"]
  B --> C["未来状态轨迹"]
  C --> D["goal-conditioned RL"]
  D --> E["统一关节目标策略"]
```

## 工程实践

| 检查项 | 建议 |
|--------|------|
| 开源 | 截至策展日项目页未见训练代码入口（待再核） |
| 频率 | 控制约 50 Hz、感知/生成约 20 Hz 的异步部署 |
| 对照 | 与去噪世界模型 loco（P017）分开比较问题接口 |

## 源码运行时序图

**不适用**（项目页未见稳定训练入口时不画伪时序图）。

## 实验与评测读法

- 关注未见地形相对蒸馏基线的提升，以及生成延迟是否进入内环。
- 固定专家数据预算做消融，才能说明世界模型而非数据拼接在起作用。

## 结论

**DreamPolicy 的要点是「生成状态参考 + 统一跟踪」，不是消灭前期专家。**

- 专家与奖励仍在数据采集阶段存在。
- 异步生成必须可丢弃重规划。
- 本页是 arXiv:2505.18780 唯一节点。

## 局限与风险

- 自回归生成误差会累积。
- 开源边界可能变化，部署前再核项目页。

## 关联页面

- [Denoising World Model Locomotion](./paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md)
- [World Action Models](../concepts/world-action-models.md)
- [HMI 论文导读](../queries/hmi-papers-coverage.md)

## 参考来源

- [hmi_p018_dreampolicy-humanoid-locomotion.md](../../sources/papers/hmi_p018_dreampolicy-humanoid-locomotion.md)
- [humanoid_pnb_one-policy-but-many-worlds-a-scalable-unified-po.md](../../sources/papers/humanoid_pnb_one-policy-but-many-worlds-a-scalable-unified-po.md)
- [humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)

## 推荐继续阅读

- [arXiv:2505.18780](https://arxiv.org/abs/2505.18780)
- [项目页](https://dreampolicy.github.io/)
- [HMI P018](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P018.md)
