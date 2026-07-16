---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bigai, sdu, tsinghua]
status: complete
updated: 2026-07-16
venue: curated
summary: "SafeFall 做的是 protective control for humanoid robots。它的出发点非常现实：双足机器人不可避免会摔倒，而摔倒会损伤传感器、执行器和结构件。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_41_safefall_learning_protective_control_for_humanoi.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# SafeFall

**SafeFall** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 41/42** 篇，归类为 **05 接触 · 柔顺 · 安全恢复**。

## 一句话定义

SafeFall 做的是 protective control for humanoid robots。它的出发点非常现实：双足机器人不可避免会摔倒，而摔倒会损伤传感器、执行器和结构件。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **05 接触 · 柔顺 · 安全恢复**（#41/42）。
- SafeFall 做的是 protective control for humanoid robots。它的出发点非常现实：双足机器人不可避免会摔倒，而摔倒会损伤传感器、执行器和结构件。
- SafeFall 不是让机器人永远不摔，而是在检测到跌倒不可避免时，激活保护策略，减少硬件冲击。系统包含一个轻量 GRU-based fall predictor 和一个 damage mitigation policy。正常控制时它保持 dormant，不干扰 nominal controller。
- 论文在 Unitree G1 上做真实实验，包括不同方向外力推扰、走路时误踩台阶、高速跑步绊倒等场景，并报告最大关节力、接触力等指标改善。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 41/42 |
| 系统栈层 | 05 接触 · 柔顺 · 安全恢复 |
| 机构 | 山东大学；BIGAI；清华大学 |
| 出处 | curated |
| 链接 | <https://safefall.github.io> |

## 核心机制（归纳）

### 1）策展导读要点

SafeFall 做的是 protective control for humanoid robots。它的出发点非常现实：双足机器人不可避免会摔倒，而摔倒会损伤传感器、执行器和结构件。

### 2）策展导读要点

SafeFall 不是让机器人永远不摔，而是在检测到跌倒不可避免时，激活保护策略，减少硬件冲击。系统包含一个轻量 GRU-based fall predictor 和一个 damage mitigation policy。正常控制时它保持 dormant，不干扰 nominal controller。

### 3）策展导读要点

论文在 Unitree G1 上做真实实验，包括不同方向外力推扰、走路时误踩台阶、高速跑步绊倒等场景，并报告最大关节力、接触力等指标改善。

### 4）策展导读要点

我的判断**人形机器人真实部署时，安全不是没有失败，而是失败不应造成灾难性损伤。**

## 常见误区

1. 柔顺/恢复策略要在 **接触丰富** 与 **长期稳定** 间折中，不能只看单帧姿态。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_41_safefall_learning_protective_control_for_humanoi.md](../../sources/papers/humanoid_rl_stack_41_safefall_learning_protective_control_for_humanoi.md)

## 参考来源

- [humanoid_rl_stack_41_safefall_learning_protective_control_for_humanoi.md](../../sources/papers/humanoid_rl_stack_41_safefall_learning_protective_control_for_humanoi.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：SafeFall](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/SafeFall__Learning_Protective_Control_for_Humanoid_Robots/SafeFall__Learning_Protective_Control_for_Humanoid_Robots.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
