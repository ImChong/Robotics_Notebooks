---
type: overview
tags: [agibot, world-models, simulation, category-hub, survey]
status: complete
updated: 2026-06-26
summary: "智元 2026-06 发布 · 03 世界模型 — GE-Sim 2.0 如何把动作条件世界推向可交互训练环境？"
related:
  - ./agibot-june-2026-release-technology-map.md
  - ./agibot-release-category-02-sim-training-eval.md
  - ./robot-world-models-training-loop-taxonomy.md
  - ../entities/ge-sim-2.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
---

# 智元发布分类 03：世界模型

> **图谱分类节点**：**03 世界模型**；总地图见 [智元 2026-06 发布技术地图](./agibot-june-2026-release-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| VLM | Vision-Language Model | 视觉-语言多模态大模型 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |

## 核心问题

**世界模型能否响应动作并进入训练闭环？** 智元区分 **世界动作模型**（动作如何改变世界）与 **世界模拟器**（可交互、可推演、可训练的环境）。

## 本组项目（1 个）

| # | 项目 | Wiki 实体 |
|---|------|-----------|
| 03 | GE-Sim 2.0 | [ge-sim-2.md](../entities/ge-sim-2.md) |

## 文内强调（策展）

| 能力 | 含义 |
|------|------|
| 动作条件建模 | 动作后环境须随之变化 |
| Eval / RL / Teleop in WM | 在世界模型内评测、强化学习、遥操作 |
| 通用奖励模型 | 文本评估生成状态，推进训练闭环 |
| 谨慎态度 | 训练场能否替代足够多真机交互 **仍待迁移验证** |

## 与 Genie Sim 3.0 分工

| 项目 | 侧重 |
|------|------|
| Genie Sim 3.0 | **刚体/场景级** 仿真训练与 benchmark |
| GE-Sim 2.0 | **视频世界模型** 闭环 rollout 与内置 critic |

## 关联页面

- [世界模型训练闭环 taxonomy](./robot-world-models-training-loop-taxonomy.md)
- [仿真训练与评测](./agibot-release-category-02-sim-training-eval.md)

## 参考来源

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)
- [ge_sim_2_arxiv_2605_27491.md](../../sources/papers/ge_sim_2_arxiv_2605_27491.md)

## 推荐继续阅读

- [GE-Sim 2.0 项目页](https://ge-sim-v2.github.io/)
