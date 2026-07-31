---

type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control, snu]
status: complete
updated: 2026-07-16
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

## 结论

**SafeWBC 把安全做成挂在 WBC/策略输出之后的一层「控制量修正器」，而不是重新训练一个更安全的策略——好处是与上游控制器解耦，代价是它只能改动作、改不了意图。**

- 真正起作用的是 **输入到状态安全控制屏障函数** 对控制量的在线修正：关节限位、自碰撞、障碍距离被统一写成状态约束，并显式留出扰动裕度。
- 接口位置是这条工作的核心取舍：安全层接在全身控制命令之后，因此对上游是模型法 WBC 还是 RL 策略都不敏感，可作为既有栈的加装件。
- 适用边界：本条目属运动小脑的 **身体层**（D 全身跟踪基座），只保障控制量层面的约束满足，不承担 VLA/世界模型那一层的任务规划与语义安全。
- 本页是 64 篇 survey 的策展编译（#36/64），机构信息（首尔大学 朴在亨课题组）自标为待最终核对；量化 benchmark 与实机指标请回到项目页与原文，不要拿本页当性能依据。

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
