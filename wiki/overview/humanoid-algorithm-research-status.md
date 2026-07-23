---
type: overview
tags: [humanoid, research, reinforcement-learning, locomotion, vla, navigation]
status: complete
updated: 2026-07-23
related:
  - ./humanoid-robot-history.md
  - ../roadmaps/humanoid-control-roadmap.md
  - ./topic-locomotion.md
  - ./large-model-empowered-humanoids.md
  - ../entities/humanoid-system-curriculum.md
  - ../tasks/humanoid-locomotion.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
summary: "人形算法研究现状鸟瞰：运动控制（模型基/RL/模仿）、loco-manipulation、导航 SLAM、感知足球与 VLA/VLN 大模型线，并指向本库更深路线图。"
---

# 人形机器人算法研究现状

## 一句话定义

**人形算法研究现状**是对当前人形研究主战场的分层快照：**下肢运动**、**全身 loco-manip**、**导航与探索**、**比赛级感知决策**、**大模型具身**——对应深蓝课程第 1.2 节，并接到本库可继续深挖的路线页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 高动态运动主流学习范式 |
| IL | Imitation Learning | 动捕/视频驱动的模仿与跟踪 |
| VLA | Vision-Language-Action | 视觉–语言–动作大模型策略 |
| VLN | Vision-Language Navigation | 语言引导导航 |
| SLAM | Simultaneous Localization and Mapping | 建图定位，支撑自主移动 |
| AMP | Adversarial Motion Priors | 对抗运动先验，人形技能常用 |

## 为什么重要

- 防止把「人形研究」窄化为单一 PPO 行走；课程后续章节正是按 **控制 → 导航 → 探索 → 感知 → 大模型** 展开的研究现状切片。
- 便于选型：做比赛足球、做 VLN、做全身操作，入口不同。

## 核心原理（现状分层）

| 层次 | 近期主流 | 本库入口 |
|------|----------|----------|
| 双足/全身运动 | PPO、AMP、运动跟踪、BFM | [Humanoid Locomotion](../tasks/humanoid-locomotion.md)、[控制路线图](../roadmaps/humanoid-control-roadmap.md) |
| 移动操作 | 分层/统一策略、接触力课程 | [Loco-Manipulation](../tasks/loco-manipulation.md) |
| 导航自主 | LiDAR SLAM、A\*/DWA、探索规划 | [导航栈](./navigation-slam-autonomy-stack.md)、[自主探索](../tasks/autonomous-exploration.md) |
| 比赛感知决策 | YOLO + 场地线几何 + EKF | [Humanoid Soccer](../tasks/humanoid-soccer.md) |
| 大模型具身 | VLA / VLN / 语音 agent | [大模型赋能人形](./large-model-empowered-humanoids.md) |

## 工程实践

- 用课程大纲当「研究地图」：Ch2–8 各对应上表一行，先建立坐标再读单篇论文。
- 跟踪综述入口：具身智能研究室人形 RL 长文等（见参考来源）。

## 局限与风险

- 「现状」更新极快，本页只做分层导航，细节以专题 overview / 论文实体为准。
- 商业演示 ≠ 可复现算法；以开源与论文评测为准。

## 关联页面

- [人形发展历史](./humanoid-robot-history.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)
- [运动控制专题](./topic-locomotion.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)
- [具身智能研究室人形 RL 运动控制长文](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)

## 推荐继续阅读

- [人形控制学习路线图](../roadmaps/humanoid-control-roadmap.md)
