---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, nvidia, cmu]
status: complete
updated: 2026-07-16
venue: curated
summary: "ASAP 的完整思想是 Aligning Simulation and Real Physics。它关注敏捷全身动作在仿真和真实之间的动力学偏差。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_25_asap_aligning_simulation_and_real_world_physics.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# ASAP

**ASAP** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 25/42** 篇，归类为 **03 感知式高动态运动**。

## 一句话定义

ASAP 的完整思想是 Aligning Simulation and Real Physics。它关注敏捷全身动作在仿真和真实之间的动力学偏差。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **03 感知式高动态运动**（#25/42）。
- ASAP 的完整思想是 Aligning Simulation and Real Physics。它关注敏捷全身动作在仿真和真实之间的动力学偏差。
- 1在仿真中用人类动作数据预训练 motion tracking policies；
- 3基于真实数据训练 delta action/model，修正仿真状态和真实状态之间的偏差。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 25/42 |
| 系统栈层 | 03 感知式高动态运动 |
| 机构 | CMU；NVIDIA |
| 出处 | curated |
| 链接 | <https://agile.human2humanoid.com> |

## 核心机制（归纳）

### 1）策展导读要点

ASAP 的完整思想是 Aligning Simulation and Real Physics。它关注敏捷全身动作在仿真和真实之间的动力学偏差。

### 2）策展导读要点

1在仿真中用人类动作数据预训练 motion tracking policies；

### 3）策展导读要点

3基于真实数据训练 delta action/model，修正仿真状态和真实状态之间的偏差。

### 4）策展导读要点

ASAP 有一点非常现实：论文直接提到真实机器人上采集敏捷动作数据会受到硬件限制，比如电机过热、硬件损伤、数据规模受限等。这不像很多 sim-to-real 论文只强调算法优雅，它承认真机高动态动作本身就是昂贵且危险的。

## 常见误区

1. 感知 locomotion 的难点在 **闭环时延与几何误差**，不是单纯「加相机输入」。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_25_asap_aligning_simulation_and_real_world_physics.md](../../sources/papers/humanoid_rl_stack_25_asap_aligning_simulation_and_real_world_physics.md)

## 参考来源

- [humanoid_rl_stack_25_asap_aligning_simulation_and_real_world_physics.md](../../sources/papers/humanoid_rl_stack_25_asap_aligning_simulation_and_real_world_physics.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
