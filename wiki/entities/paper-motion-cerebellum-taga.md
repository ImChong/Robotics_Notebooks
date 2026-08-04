---

type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control, nus, scut, zju]
status: complete
updated: 2026-07-16
venue: curated
summary: "底座：主动凝视进入敏捷运动闭环。输入是视觉场景、本体状态和任务目标；实现上学习主动凝视策略，让机器人在运动中选择看哪里，再把感知结果用于敏捷运动控制；它把视觉注意力纳入 locomotion 闭环。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-01-locomotion-base.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_09_taga.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# TAGA

**TAGA** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 09/64** 篇，归类为 **A 走路底座**。

## 一句话定义

底座：主动凝视进入敏捷运动闭环。输入是视觉场景、本体状态和任务目标；实现上学习主动凝视策略，让机器人在运动中选择看哪里，再把感知结果用于敏捷运动控制；它把视觉注意力纳入 locomotion 闭环。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 底座：主动凝视进入敏捷运动闭环。输入是视觉场景、本体状态和任务目标；实现上学习主动凝视策略，让机器人在运动中选择看哪里，再把感知结果用于敏捷运动控制；它把视觉注意力纳入 locomotion 闭环。
- 运动小脑 64 篇 **#09/64** · 底座：主动凝视进入敏捷运动闭环。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 09/64 |
| 分组 | A 走路底座 |
| 机构 | 新加坡国立大学 Marmot 实验室、浙江大学 X-力学中心、华南理工大学 |
| 论文/项目 | https://marmotlab.github.io/taga-humanoid/ |

## 核心机制（归纳）

### 1）策展导读要点

底座：主动凝视进入敏捷运动闭环。输入是视觉场景、本体状态和任务目标；实现上学习主动凝视策略，让机器人在运动中选择看哪里，再把感知结果用于敏捷运动控制；它把视觉注意力纳入 locomotion 闭环。

### 2）策展导读要点

机构：新加坡国立大学 Marmot 实验室、浙江大学 X-力学中心、华南理工大学

## 结论

**TAGA 改的是感知与运动之间的接口：让「看哪里」本身成为一个被学习的策略，和敏捷运动共处同一个闭环，而不是把视觉当成外部给定的输入流。**

- 关键机制是 **主动凝视策略**：机器人在运动过程中自行决定视线落点，感知结果再回灌到敏捷运动控制，视觉注意力由此成为 locomotion 闭环内的可控变量。
- 归在 **A 走路底座**（#09/64），提供的是底座层能力；输入同时吃视觉场景、本体状态与任务目标，意味着它假定下游已有可用的视觉与本体感知栈。
- 适用边界：解决的是身体层问题，不替代 VLA / 世界模型的任务规划。
- 本页为策展编译的索引级归纳，量化 benchmark、消融与实机指标以 [参考来源](#参考来源) 中的原文与项目页为准。

## 常见误区

1. 运动小脑条目解决 **身体层** 问题，不替代 VLA/世界模型的任务规划。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-01-locomotion-base.md](../overview/motion-cerebellum-category-01-locomotion-base.md)

## 参考来源

- [motion_cerebellum_survey_09_taga.md](../../sources/papers/motion_cerebellum_survey_09_taga.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
