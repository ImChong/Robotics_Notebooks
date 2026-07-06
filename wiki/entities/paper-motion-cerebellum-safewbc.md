---

type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control, snu]
status: complete
updated: 2026-07-06
venue: curated
summary: "安全：控制屏障函数接到全身控制后面。输入是全身控制命令、状态约束和扰动裕度；实现上用输入到状态安全控制屏障函数修正控制量，使关节限位、自碰撞、障碍距离等约束保持安全；重点是把安全层接到 WBC/策略输出后面。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_36_safewbc.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# SafeWBC

**SafeWBC** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 36/64** 篇，归类为 **D 全身跟踪基座**。

## 一句话定义

安全：控制屏障函数接到全身控制后面。输入是全身控制命令、状态约束和扰动裕度；实现上用输入到状态安全控制屏障函数修正控制量，使关节限位、自碰撞、障碍距离等约束保持安全；重点是把安全层接到 WBC/策略输出后面。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 安全：控制屏障函数接到全身控制后面。输入是全身控制命令、状态约束和扰动裕度；实现上用输入到状态安全控制屏障函数修正控制量，使关节限位、自碰撞、障碍距离等约束保持安全；重点是把安全层接到 WBC/策略输出后面。
- 运动小脑 64 篇 **#36/64** · 安全：控制屏障函数接到全身控制后面。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 36/64 |
| 分组 | D 全身跟踪基座 |
| 机构 | 首尔大学 朴在亨课题组（待最终核对） |
| 论文/项目 | https://kwlee365.github.io/SafeWBC-Website/ |

## 核心机制（归纳）

### 1）策展导读要点

安全：控制屏障函数接到全身控制后面。输入是全身控制命令、状态约束和扰动裕度；实现上用输入到状态安全控制屏障函数修正控制量，使关节限位、自碰撞、障碍距离等约束保持安全；重点是把安全层接到 WBC/策略输出后面。

### 2）策展导读要点

机构：首尔大学 朴在亨课题组（待最终核对）

## 常见误区

1. 运动小脑条目解决 **身体层** 问题，不替代 VLA/世界模型的任务规划。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-04-wbt-base.md](../overview/motion-cerebellum-category-04-wbt-base.md)

## 参考来源

- [motion_cerebellum_survey_36_safewbc.md](../../sources/papers/motion_cerebellum_survey_36_safewbc.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
