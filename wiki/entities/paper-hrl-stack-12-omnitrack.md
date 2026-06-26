---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, sjtu, hust, bigai]
status: complete
updated: 2026-06-26
venue: curated
summary: "OmniTrack 关注 physics-consistent reference。它的出发点是：从人类动作或重定向数据里得到的参考轨迹常常不干净，可能有浮空、脚滑、不稳定接触和噪声。如果训练策略时强迫控制器去追踪这些参考，策略就会在“像参考”和“保持物理稳定”之间冲突。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# OmniTrack

**OmniTrack** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 12/42** 篇，归类为 **02 参考跟踪 · 通用控制**。

## 一句话定义

OmniTrack 关注 physics-consistent reference。它的出发点是：从人类动作或重定向数据里得到的参考轨迹常常不干净，可能有浮空、脚滑、不稳定接触和噪声。如果训练策略时强迫控制器去追踪这些参考，策略就会在“像参考”和“保持物理稳定”之间冲突。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **02 参考跟踪 · 通用控制**（#12/42）。
- OmniTrack 关注 physics-consistent reference。它的出发点是：从人类动作或重定向数据里得到的参考轨迹常常不干净，可能有浮空、脚滑、不稳定接触和噪声。如果训练策略时强迫控制器去追踪这些参考，策略就会在“像参考”和“保持物理稳定”之间冲突。
- OmniTrack 的思路是解耦：先由 privileged generalist policy 生成更严格满足物理约束的参考，再训练通用跟踪器。也就是说，不是直接追踪 raw retargeted motions，而是先把参考轨迹变得更物理一致。
- 这篇论文的意义在于，它把 reference quality 放在了 motion tracking 系统中心。它不是让控制器无限背锅，而是先清理参考本身。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 12/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 机构 | 华中科技大学；BIGAI；上海交通大学 |
| 出处 | curated |
| 链接 | <https://omnitrack-humanoid.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

OmniTrack 关注 physics-consistent reference。它的出发点是：从人类动作或重定向数据里得到的参考轨迹常常不干净，可能有浮空、脚滑、不稳定接触和噪声。如果训练策略时强迫控制器去追踪这些参考，策略就会在“像参考”和“保持物理稳定”之间冲突。

### 2）策展导读要点

OmniTrack 的思路是解耦：先由 privileged generalist policy 生成更严格满足物理约束的参考，再训练通用跟踪器。也就是说，不是直接追踪 raw retargeted motions，而是先把参考轨迹变得更物理一致。

### 3）策展导读要点

这篇论文的意义在于，它把 reference quality 放在了 motion tracking 系统中心。它不是让控制器无限背锅，而是先清理参考本身。

### 4）策展导读要点

我的判断**未来通用 motion tracker 的核心能力之一，不是“什么都追”，而是知道什么参考值得追、怎么把参考变成可追。**

## 常见误区

1. Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md](../../sources/papers/humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md)

## 参考来源

- [humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md](../../sources/papers/humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
