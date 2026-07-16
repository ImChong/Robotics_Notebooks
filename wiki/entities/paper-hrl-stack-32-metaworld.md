---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bjut, fudan, tsinghua]
status: complete
updated: 2026-07-16
arxiv: "2601.17507"
venue: "arXiv"
summary: "MetaWorld 是一个 hierarchical world model。它把系统分成三层：semantic layer、skill transfer layer、physical layer。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_32_metaworld_skill_transfer_and_composition_in_a_hi.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# MetaWorld

**MetaWorld** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 32/42** 篇，归类为 **04 视觉闭环 · 任务接口 · 世界模型**。

## 一句话定义

MetaWorld 是一个 hierarchical world model。它把系统分成三层：semantic layer、skill transfer layer、physical layer。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#32/42）。
- MetaWorld 是一个 hierarchical world model。它把系统分成三层：semantic layer、skill transfer layer、physical layer。
- 语义层用 GPT-4o 或 VLM 解析高层指令，输出技能权重；技能迁移层做动态专家选择；物理层用 TD-MPC2 和专家动作引导完成控制。
- 这篇论文没有真实机器人实验，主要在 HumanoidBench 中验证。但它适合放在这里，因为它代表了一种不同于 SENTINEL 的路线。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 32/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 机构 | 北京工业大学；复旦大学；清华大学 |
| 出处 | arXiv |
| 链接 | <https://arxiv.org/abs/2601.17507> |

## 核心机制（归纳）

### 1）策展导读要点

MetaWorld 是一个 hierarchical world model。它把系统分成三层：semantic layer、skill transfer layer、physical layer。

### 2）策展导读要点

语义层用 GPT-4o 或 VLM 解析高层指令，输出技能权重；技能迁移层做动态专家选择；物理层用 TD-MPC2 和专家动作引导完成控制。

### 3）策展导读要点

这篇论文没有真实机器人实验，主要在 HumanoidBench 中验证。但它适合放在这里，因为它代表了一种不同于 SENTINEL 的路线。

### 4）策展导读要点

我的判断**未来不会是纯端到端或纯模块化的二选一，更可能是“可学习的模块化”：每一层都学习，但接口保持清楚。**

## 常见误区

1. VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_32_metaworld_skill_transfer_and_composition_in_a_hi.md](../../sources/papers/humanoid_rl_stack_32_metaworld_skill_transfer_and_composition_in_a_hi.md)

## 参考来源

- [humanoid_rl_stack_32_metaworld_skill_transfer_and_composition_in_a_hi.md](../../sources/papers/humanoid_rl_stack_32_metaworld_skill_transfer_and_composition_in_a_hi.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
