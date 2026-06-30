---

type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control, hit, leju]
status: complete
updated: 2026-06-30
venue: curated
summary: "底座：把导航接口接到地形自适应步态。输入是导航速度/路径参考、地形几何和机器人状态；实现上把参考轨迹投到可落脚地形上，再训练低层策略跟踪 SE(2) 速度接口；这样标准导航栈可以调用人形步态。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-01-locomotion-base.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_01_guidewalk.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# GuideWalk

**GuideWalk** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 01/64** 篇，归类为 **A 走路底座**。

## 一句话定义

底座：把导航接口接到地形自适应步态。输入是导航速度/路径参考、地形几何和机器人状态；实现上把参考轨迹投到可落脚地形上，再训练低层策略跟踪 SE(2) 速度接口；这样标准导航栈可以调用人形步态。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 底座：把导航接口接到地形自适应步态。输入是导航速度/路径参考、地形几何和机器人状态；实现上把参考轨迹投到可落脚地形上，再训练低层策略跟踪 SE(2) 速度接口；这样标准导航栈可以调用人形步态。
- 运动小脑 64 篇 **#01/64** · 底座：把导航接口接到地形自适应步态。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 01/64 |
| 分组 | A 走路底座 |
| 机构 | 哈尔滨工业大学、乐聚机器人 |
| 论文/项目 | https://GuideWalk.github.io |

## 核心机制（归纳）

### 1）策展导读要点

底座：把导航接口接到地形自适应步态。输入是导航速度/路径参考、地形几何和机器人状态；实现上把参考轨迹投到可落脚地形上，再训练低层策略跟踪 SE(2) 速度接口；这样标准导航栈可以调用人形步态。

### 2）策展导读要点

机构：哈尔滨工业大学、乐聚机器人

## 常见误区

1. 运动小脑条目解决 **身体层** 问题，不替代 VLA/世界模型的任务规划。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-01-locomotion-base.md](../overview/motion-cerebellum-category-01-locomotion-base.md)

## 参考来源

- [motion_cerebellum_survey_01_guidewalk.md](../../sources/papers/motion_cerebellum_survey_01_guidewalk.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
