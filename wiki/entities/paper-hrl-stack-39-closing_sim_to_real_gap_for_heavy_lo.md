---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, lumos, teleai, sjtu, zju]
status: complete
updated: 2026-07-06
venue: curated
summary: "HALO 解决 heavy-loaded humanoid agile motion skills。真实机器人执行任务时经常会携带未知负载，而负载会改变机器人动力学。一个空载时表现很好的策略，拿重物后可能直接失效。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-08-real-tasks.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_39_closing_sim_to_real_gap_for_heavy_loaded_humanoi.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# Closing Sim-to-Real Gap for Heavy-loaded Humanoid Agile Motion Skills via Differentiable Simulation

**Closing Sim-to-Real Gap for Heavy-loaded Humanoid Agile Motion Skills via Differentiable Simulation** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 39/42** 篇，归类为 **05 接触 · 柔顺 · 安全恢复**。

## 一句话定义

HALO 解决 heavy-loaded humanoid agile motion skills。真实机器人执行任务时经常会携带未知负载，而负载会改变机器人动力学。一个空载时表现很好的策略，拿重物后可能直接失效。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **05 接触 · 柔顺 · 安全恢复**（#39/42）。
- HALO 解决 heavy-loaded humanoid agile motion skills。真实机器人执行任务时经常会携带未知负载，而负载会改变机器人动力学。一个空载时表现很好的策略，拿重物后可能直接失效。
- HALO 用 differentiable simulation 做两阶段系统辨识。第一阶段校准名义机器人模型，减少固有 sim-to-real 差异；第二阶段识别 payload mass distribution，处理带负载后的动力学变化。
- 论文的实验包括真实机器人上的高动态动作和负载情况。它强调，通过显式减少结构化模型误差，可以让 RL 策略在重载条件下更稳定地零样本迁移。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 39/42 |
| 系统栈层 | 05 接触 · 柔顺 · 安全恢复 |
| 机构 | 浙江大学；中国电信 TeleAI；上海交通大学；Lumos Robotics |
| 出处 | curated |
| 链接 | <https://mwondering.github.io/halo-humanoid/> |

## 核心机制（归纳）

### 1）策展导读要点

HALO 解决 heavy-loaded humanoid agile motion skills。真实机器人执行任务时经常会携带未知负载，而负载会改变机器人动力学。一个空载时表现很好的策略，拿重物后可能直接失效。

### 2）策展导读要点

HALO 用 differentiable simulation 做两阶段系统辨识。第一阶段校准名义机器人模型，减少固有 sim-to-real 差异；第二阶段识别 payload mass distribution，处理带负载后的动力学变化。

### 3）策展导读要点

论文的实验包括真实机器人上的高动态动作和负载情况。它强调，通过显式减少结构化模型误差，可以让 RL 策略在重载条件下更稳定地零样本迁移。

### 4）策展导读要点

我的判断**人形机器人一旦进入搬运和工具使用场景，空载控制器就不够了，负载辨识会成为基础能力。**

## 常见误区

1. 柔顺/恢复策略要在 **接触丰富** 与 **长期稳定** 间折中，不能只看单帧姿态。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_39_closing_sim_to_real_gap_for_heavy_loaded_humanoi.md](../../sources/papers/humanoid_rl_stack_39_closing_sim_to_real_gap_for_heavy_loaded_humanoi.md)

## 参考来源

- [humanoid_rl_stack_39_closing_sim_to_real_gap_for_heavy_loaded_humanoi.md](../../sources/papers/humanoid_rl_stack_39_closing_sim_to_real_gap_for_heavy_loaded_humanoi.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
