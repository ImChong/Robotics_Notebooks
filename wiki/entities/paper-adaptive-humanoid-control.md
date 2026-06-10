---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, amp, motion-prior]
status: complete
updated: 2026-06-10
venue: curated
summary: "AHC：multi-behavior distillation + reinforced fine-tuning 训练统一人形控制器；在 RL 身体系统栈与 AMP 专题均属多技能/通用控制簇。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_21_towards_adaptive_humanoid_control_via_multi_beha.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/humanoid_amp_survey_11_towards_adaptive_humanoid_control_via_multi_beha.md
  - ../../sources/papers/humanoid_amp_survey_19_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md
---

# Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning

**Towards Adaptive Humanoid Control**（AHC）通过 multi-behavior distillation 与 reinforced fine-tuning 训练统一控制器：先训练多个基础行为策略，再蒸馏为 multi-behavior controller，最后用强化微调提升地形适应与跌倒恢复。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |
| AHC | Adaptive Humanoid Control | 多行为蒸馏与强化微调统一的人形自适应控制框架 |

## 为什么重要

- 处理**多行为统一控制**：单一策略覆盖走、跑、恢复等多种基础行为。
- 蒸馏 + 强化微调组合，兼顾行为多样性与地形适应能力。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 项目页 | <https://ahc-humanoid.github.io> |
| 机构 | 哈尔滨工程大学；中国电信 TeleAI；中科大；上海科技大学；哈尔滨工业大学；西北工业大学深圳研究院 |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 21/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 AMP 19 篇运动先验专题中

| 字段 | 内容 |
|------|------|
| 编号 | 11/19 |
| 叙事段 | 03 多技能与自适应 |
| 索引来源 | [具身智能研究室 · AMP 运动先验专题](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w) |

## 与其他页面的关系

- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 综述：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息** 表）。
- 若需与姊妹篇对照，请回到对应 **技术地图 / 42 篇栈 / AMP 专题** 总览中的实验段落。

## 参考来源

- [humanoid_rl_stack_21_towards_adaptive_humanoid_control_via_multi_beha.md](../../sources/papers/humanoid_rl_stack_21_towards_adaptive_humanoid_control_via_multi_beha.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [humanoid_amp_survey_11_towards_adaptive_humanoid_control_via_multi_beha.md](../../sources/papers/humanoid_amp_survey_11_towards_adaptive_humanoid_control_via_multi_beha.md) — AMP 专题策展摘录
- [humanoid_amp_survey_19_catalog.md](../../sources/papers/humanoid_amp_survey_19_catalog.md) — AMP 19 篇总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md) — AMP 专题微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)、[wechat_humanoid_amp_19_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_amp_19_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [AMP 专题长文（微信公众号）](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
