---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, ningbo-eastern, polyu, limx, ustc]
status: complete
updated: 2026-06-25
venue: "project"
code: https://github.com/myismyname/SRL4Humanoid
summary: "PvP 的全称是 Proprioceptive-Privileged Contrastive Representations。它关注的是 whole-body control 中的样本效率和部分可观测问题。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_20_pvp_data_efficient_humanoid_robot_learning_with.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# PvP

**PvP** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 20/42** 篇，归类为 **02 参考跟踪 · 通用控制**。

## 一句话定义

PvP 的全称是 Proprioceptive-Privileged Contrastive Representations。它关注的是 whole-body control 中的样本效率和部分可观测问题。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **02 参考跟踪 · 通用控制**（#20/42）。
- PvP 的全称是 Proprioceptive-Privileged Contrastive Representations。它关注的是 whole-body control 中的样本效率和部分可观测问题。
- 真实机器人部署时能用的通常是 proprioception，例如关节角、关节速度、IMU、历史动作等；但仿真训练时可以获得更完整的 privileged states，例如身体速度、接触状态、地形信息等。PvP 试图利用二者之间的互补关系，通过对比学习得到紧凑、任务相关的 latent representation。
- 这篇论文不一定像跑酷、开门那样有很强的视频冲击力，但它解决的是很多控制器的基础问题：**怎么把训练时看得到、部署时看不到的信息，变成部署时仍然有用的表征。**

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 20/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 机构 | 香港理工大学；逐际动力；宁波东方理工大学；中科大；ZJU-UIUC；ZGCA |
| 出处 | project |
| 链接 | <https://github.com/myismyname/SRL4Humanoid> |

## 核心机制（归纳）

### 1）策展导读要点

PvP 的全称是 Proprioceptive-Privileged Contrastive Representations。它关注的是 whole-body control 中的样本效率和部分可观测问题。

### 2）策展导读要点

真实机器人部署时能用的通常是 proprioception，例如关节角、关节速度、IMU、历史动作等；但仿真训练时可以获得更完整的 privileged states，例如身体速度、接触状态、地形信息等。PvP 试图利用二者之间的互补关系，通过对比学习得到紧凑、任务相关的 latent representation。

### 3）策展导读要点

这篇论文不一定像跑酷、开门那样有很强的视频冲击力，但它解决的是很多控制器的基础问题：**怎么把训练时看得到、部署时看不到的信息，变成部署时仍然有用的表征。**

### 4）策展导读要点

我的判断**这种 proprioceptive-privileged 表示学习会越来越常见，因为它正好连接了仿真训练和真实部署之间的信息落差。**

## 常见误区

1. Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_20_pvp_data_efficient_humanoid_robot_learning_with.md](../../sources/papers/humanoid_rl_stack_20_pvp_data_efficient_humanoid_robot_learning_with.md)

## 参考来源

- [humanoid_rl_stack_20_pvp_data_efficient_humanoid_robot_learning_with.md](../../sources/papers/humanoid_rl_stack_20_pvp_data_efficient_humanoid_robot_learning_with.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
