---

type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control, sysu]
status: complete
updated: 2026-07-16
venue: curated
summary: "接口：上层规划与泛化动作小脑分工。输入是上层任务命令、物体/负载状态和机器人本体状态；实现上把移动、操作、平衡和负载扰动拆成可控接口，再通过蒸馏、MPC 引导或强化学习训练全身策略；重点是让 VLA/planner 可以稳定调用身体。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-07-loco-manip-interface.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_51_active_spatial_brain_generalized_cerebellum.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 主动空间大脑与泛化动作小脑

**主动空间大脑与泛化动作小脑** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 51/64** 篇，归类为 **G Loco-Manip 接口**。

## 一句话定义

接口：上层规划与泛化动作小脑分工。输入是上层任务命令、物体/负载状态和机器人本体状态；实现上把移动、操作、平衡和负载扰动拆成可控接口，再通过蒸馏、MPC 引导或强化学习训练全身策略；重点是让 VLA/planner 可以稳定调用身体。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 接口：上层规划与泛化动作小脑分工。输入是上层任务命令、物体/负载状态和机器人本体状态；实现上把移动、操作、平衡和负载扰动拆成可控接口，再通过蒸馏、MPC 引导或强化学习训练全身策略；重点是让 VLA/planner 可以稳定调用身体。
- 运动小脑 64 篇 **#51/64** · 接口：上层规划与泛化动作小脑分工。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 51/64 |
| 分组 | G Loco-Manip 接口 |
| 机构 | 中山大学计算机科学与工程学院 |
| 论文/项目 | https://leungchaos.github.io/Humanoid-Whole-Body-Manipulation-via-Active- |

## 核心机制（归纳）

### 1）策展导读要点

接口：上层规划与泛化动作小脑分工。输入是上层任务命令、物体/负载状态和机器人本体状态；实现上把移动、操作、平衡和负载扰动拆成可控接口，再通过蒸馏、MPC 引导或强化学习训练全身策略；重点是让 VLA/planner 可以稳定调用身体。

### 2）策展导读要点

机构：中山大学计算机科学与工程学院

## 结论

**这条目的价值不在于某个动作学得更好，而在于把「上层规划怎么稳定调用身体」定义成一个接口问题：移动、操作、平衡、负载扰动各自拆成可控入口。**

- 起作用的是拆分方式本身，训练手段（蒸馏、MPC 引导或强化学习）可替换；接口稳定了，VLA/planner 才有可调用的对象。
- 输入侧显式包含物体/负载状态，说明它把负载扰动当成一等输入建模，而不是当噪声去抑制。
- 边界清晰：这是身体层的解，不替代 VLA/世界模型的任务规划；上层给不出合理命令时，接口再干净也无能为力。
- 本页为策展索引级，未记录发表日期与量化指标；量化 benchmark、消融与实机数据以原文 PDF / 项目页为准。

## 常见误区

1. 运动小脑条目解决 **身体层** 问题，不替代 VLA/世界模型的任务规划。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-07-loco-manip-interface.md](../overview/motion-cerebellum-category-07-loco-manip-interface.md)

## 参考来源

- [motion_cerebellum_survey_51_active_spatial_brain_generalized_cerebellum.md](../../sources/papers/motion_cerebellum_survey_51_active_spatial_brain_generalized_cerebellum.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
