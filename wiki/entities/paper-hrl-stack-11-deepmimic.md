---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, ubc, berkeley]
status: complete
updated: 2026-07-06
venue: curated
summary: "DeepMimic 是这批论文里最经典的一篇。它来自物理角色动画领域，但今天再看，它仍然是很多 humanoid motion tracking 工作的起点。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-02-motion-imitation.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../methods/deepmimic.md
sources:
  - ../../sources/papers/humanoid_rl_stack_11_deepmimic_example_guided_deep_reinforcement_lear.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# DeepMimic

**DeepMimic** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 11/42** 篇，归类为 **02 参考跟踪 · 通用控制**。

> **深读页：** [deepmimic](../methods/deepmimic.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

DeepMimic 是这批论文里最经典的一篇。它来自物理角色动画领域，但今天再看，它仍然是很多 humanoid motion tracking 工作的起点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **02 参考跟踪 · 通用控制**（#11/42）。
- DeepMimic 是这批论文里最经典的一篇。它来自物理角色动画领域，但今天再看，它仍然是很多 humanoid motion tracking 工作的起点。
- 它要解决的问题很朴素：给定一段 reference motion clip，怎么让一个物理仿真角色学会执行类似动作，同时还能满足任务目标。论文用深度强化学习，把 imitation reward 和 task reward 结合起来，训练角色完成走路、跑步、翻滚、武术、投掷等动作。
- DeepMimic 的关键贡献不是“让角色动起来”，而是提出了一种后来被大量继承的范式：

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 11/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 机构 | 伯克利；英属哥伦比亚大学 |
| 出处 | curated |
| 链接 | <https://xbpeng.github.io/projects/DeepMimic/index.html> |

## 核心机制（归纳）

### 1）策展导读要点

DeepMimic 是这批论文里最经典的一篇。它来自物理角色动画领域，但今天再看，它仍然是很多 humanoid motion tracking 工作的起点。

### 2）策展导读要点

它要解决的问题很朴素：给定一段 reference motion clip，怎么让一个物理仿真角色学会执行类似动作，同时还能满足任务目标。论文用深度强化学习，把 imitation reward 和 task reward 结合起来，训练角色完成走路、跑步、翻滚、武术、投掷等动作。

### 3）策展导读要点

DeepMimic 的关键贡献不是“让角色动起来”，而是提出了一种后来被大量继承的范式：

### 4）策展导读要点

这个中间层对人形机器人非常重要。因为纯 RL 从零学复杂全身动作，探索成本太高；而纯 kinematic 动作又不保证物理可执行。DeepMimic 夹在中间：既利用人类动作数据，又让动作通过物理仿真闭环。

## 常见误区

1. Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_11_deepmimic_example_guided_deep_reinforcement_lear.md](../../sources/papers/humanoid_rl_stack_11_deepmimic_example_guided_deep_reinforcement_lear.md)

## 参考来源

- [humanoid_rl_stack_11_deepmimic_example_guided_deep_reinforcement_lear.md](../../sources/papers/humanoid_rl_stack_11_deepmimic_example_guided_deep_reinforcement_lear.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：DeepMimic](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/01_Foundational_RL/DeepMimic_Example-Guided_Deep_RL_of_Physics-Based_Character_Skills/DeepMimic_Example-Guided_Deep_RL_of_Physics-Based_Character_Skills.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
