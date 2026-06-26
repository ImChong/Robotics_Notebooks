---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, nju, huawei]
status: complete
updated: 2026-06-26
venue: curated
summary: "NMR，也就是 Neural Motion Retargeting，进一步推进了 GMR 的问题。它认为传统优化式 retargeting 是非凸的，容易出现局部最优，从而带来 self-penetration、foot sliding、物理不可行等伪影。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-03-data-pipeline.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../methods/neural-motion-retargeting-nmr.md
sources:
  - ../../sources/papers/humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# Make Tracking Easy

**Make Tracking Easy** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 02/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。

> **深读页：** [neural-motion-retargeting-nmr](../methods/neural-motion-retargeting-nmr.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

NMR，也就是 Neural Motion Retargeting，进一步推进了 GMR 的问题。它认为传统优化式 retargeting 是非凸的，容易出现局部最优，从而带来 self-penetration、foot sliding、物理不可行等伪影。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |
| GMR | General Motion Retargeting | 把人体/视频动作重定向为机器人可执行参考 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **01 数据 · 重定向 · 遥操作**（#02/42）。
- NMR，也就是 Neural Motion Retargeting，进一步推进了 GMR 的问题。它认为传统优化式 retargeting 是非凸的，容易出现局部最优，从而带来 self-penetration、foot sliding、物理不可行等伪影。
- 它的核心思路是：不要只把 retargeting 看成逐帧几何优化，而要把它看成 **数据分布学习**。论文提出 CEPR 等机制，用层次化和 VAE-based motion clustering 的方式，把大量动作组织成 latent motifs，再学习更稳定的重定向过程。
- NMR 的重要性在于，它把 **“重定向质量”** 进一步推到了模型化层面。GMR 强调 retargeting matters，NMR 则进一步问：如果传统 retargeting 本身不稳定，能不能训练一个神经网络来学习更好的 retargeting 分布？

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 02/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | 南京大学；华为 |
| 出处 | curated |
| 链接 | <https://nju3dv-humanoidgroup.github.io/nmr.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

NMR，也就是 Neural Motion Retargeting，进一步推进了 GMR 的问题。它认为传统优化式 retargeting 是非凸的，容易出现局部最优，从而带来 self-penetration、foot sliding、物理不可行等伪影。

### 2）策展导读要点

它的核心思路是：不要只把 retargeting 看成逐帧几何优化，而要把它看成 **数据分布学习**。论文提出 CEPR 等机制，用层次化和 VAE-based motion clustering 的方式，把大量动作组织成 latent motifs，再学习更稳定的重定向过程。

### 3）策展导读要点

NMR 的重要性在于，它把 **“重定向质量”** 进一步推到了模型化层面。GMR 强调 retargeting matters，NMR 则进一步问：如果传统 retargeting 本身不稳定，能不能训练一个神经网络来学习更好的 retargeting 分布？

### 4）策展导读要点

这件事对人形机器人很实际。因为未来动作来源不会只有干净 mocap，还会有 monocular video、生成视频、遥操作、互联网视频、混合数据源。数据越杂，传统优化式重定向越难完全靠手工规则解决。

## 常见误区

1. 重定向/遥操作不是「训练前脚本」——参考质量上限往往 **早于** RL 策略决定。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md](../../sources/papers/humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md)

## 参考来源

- [humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md](../../sources/papers/humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：Make Tracking Easy](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_Motion_Retargeting/Make_Tracking_Easy__Neural_Motion_Retargeting_for_Humanoid_Whole-body_Control/Make_Tracking_Easy__Neural_Motion_Retargeting_for_Humanoid_Whole-body_Control.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
