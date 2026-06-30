---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, ut-austin, nvidia, stanford]
status: complete
updated: 2026-06-30
venue: curated
summary: "CHIP 的题目是 Adaptive Compliance for Humanoid Control through Hindsight Perturbation。它想让已有 motion tracking controller 获得可调 end-effector compliance。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-09-compliance-contact.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_36_chip_adaptive_compliance_for_humanoid_control_th.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# CHIP

**CHIP** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 36/42** 篇，归类为 **05 接触 · 柔顺 · 安全恢复**。

## 一句话定义

CHIP 的题目是 Adaptive Compliance for Humanoid Control through Hindsight Perturbation。它想让已有 motion tracking controller 获得可调 end-effector compliance。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **05 接触 · 柔顺 · 安全恢复**（#36/42）。
- CHIP 的题目是 Adaptive Compliance for Humanoid Control through Hindsight Perturbation。它想让已有 motion tracking controller 获得可调 end-effector compliance。
- 传统刚性 motion tracking 在高动态动作里很好用，但在擦白板、推车、开门、协作搬运等任务里会出问题。因为这些任务要求末端执行器在受力时产生合理偏移，而不是强行保持目标位置。
- CHIP 的核心是 hindsight perturbation：训练时向末端施加扰动力，但在观测目标里事后扣除扰动偏移，让策略把受力后的偏移当成合理状态，而不是立刻强行纠正。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 36/42 |
| 系统栈层 | 05 接触 · 柔顺 · 安全恢复 |
| 机构 | NVIDIA；斯坦福大学；德州大学奥斯汀分校 |
| 出处 | curated |
| 链接 | <https://nvlabs.github.io/CHIP/> |

## 核心机制（归纳）

### 1）策展导读要点

CHIP 的题目是 Adaptive Compliance for Humanoid Control through Hindsight Perturbation。它想让已有 motion tracking controller 获得可调 end-effector compliance。

### 2）策展导读要点

传统刚性 motion tracking 在高动态动作里很好用，但在擦白板、推车、开门、协作搬运等任务里会出问题。因为这些任务要求末端执行器在受力时产生合理偏移，而不是强行保持目标位置。

### 3）策展导读要点

CHIP 的核心是 hindsight perturbation：训练时向末端施加扰动力，但在观测目标里事后扣除扰动偏移，让策略把受力后的偏移当成合理状态，而不是立刻强行纠正。

### 4）策展导读要点

它还把柔顺系数变成可调输入。比如开门阶段可能需要刚一点，擦白板需要软一点，搬重物需要根据重量调整刚度。

## 常见误区

1. 柔顺/恢复策略要在 **接触丰富** 与 **长期稳定** 间折中，不能只看单帧姿态。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_36_chip_adaptive_compliance_for_humanoid_control_th.md](../../sources/papers/humanoid_rl_stack_36_chip_adaptive_compliance_for_humanoid_control_th.md)

## 参考来源

- [humanoid_rl_stack_36_chip_adaptive_compliance_for_humanoid_control_th.md](../../sources/papers/humanoid_rl_stack_36_chip_adaptive_compliance_for_humanoid_control_th.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
