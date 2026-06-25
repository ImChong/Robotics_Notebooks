---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, nvidia]
status: complete
updated: 2026-06-25
arxiv: "2512.01061"
venue: "arXiv"
summary: "DoorMan 解决的是纯 RGB pixel-to-action 的人形开门任务。开门听起来简单，但对人形机器人来说非常复杂。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../entities/paper-doorman-opening-sim2real-door.md
sources:
  - ../../sources/papers/humanoid_rl_stack_29_opening_the_sim_to_real_door_for_humanoid_pixel.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer

**Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 29/42** 篇，归类为 **04 视觉闭环 · 任务接口 · 世界模型**。

> **深读页：** [paper-doorman-opening-sim2real-door](../entities/paper-doorman-opening-sim2real-door.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

DoorMan 解决的是纯 RGB pixel-to-action 的人形开门任务。开门听起来简单，但对人形机器人来说非常复杂。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RGB | Red-Green-Blue | 彩色图像通道，常与深度 (RGB-D) 配合 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#29/42）。
- DoorMan 解决的是纯 RGB pixel-to-action 的人形开门任务。开门听起来简单，但对人形机器人来说非常复杂。
- 机器人需要接近门、识别把手、抓住把手、旋转或推拉、跟随门板运动，同时保持全身平衡。门把手和门板是 articulated object，接触状态会随着动作变化。
- DoorMan 的方法是 teacher-student-bootstrap。第一阶段用 privileged teacher 学习完整开门流程；第二阶段蒸馏成 RGB student；第三阶段用 GRPO fine-tuning 处理残余部分可观测性差距。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 29/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 机构 | NVIDIA；伯克利；CMU；香港中文大学 |
| 出处 | arXiv |
| 链接 | <https://arxiv.org/abs/2512.01061> |

## 核心机制（归纳）

### 1）策展导读要点

DoorMan 解决的是纯 RGB pixel-to-action 的人形开门任务。开门听起来简单，但对人形机器人来说非常复杂。

### 2）策展导读要点

机器人需要接近门、识别把手、抓住把手、旋转或推拉、跟随门板运动，同时保持全身平衡。门把手和门板是 articulated object，接触状态会随着动作变化。

### 3）策展导读要点

DoorMan 的方法是 teacher-student-bootstrap。第一阶段用 privileged teacher 学习完整开门流程；第二阶段蒸馏成 RGB student；第三阶段用 GRPO fine-tuning 处理残余部分可观测性差距。

### 4）策展导读要点

DoorMan 最值得注意的是 staged-reset exploration。长时程开门任务如果从起点随机探索，策略很难经常到达关键中间状态。阶段重置把训练起点放到不同任务阶段，让策略能集中学习抓握、旋转、推拉等后半段难点。

## 常见误区

1. VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_29_opening_the_sim_to_real_door_for_humanoid_pixel.md](../../sources/papers/humanoid_rl_stack_29_opening_the_sim_to_real_door_for_humanoid_pixel.md)

## 参考来源

- [humanoid_rl_stack_29_opening_the_sim_to_real_door_for_humanoid_pixel.md](../../sources/papers/humanoid_rl_stack_29_opening_the_sim_to_real_door_for_humanoid_pixel.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
