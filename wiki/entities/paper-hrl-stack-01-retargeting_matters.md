---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, stanford]
status: complete
updated: 2026-07-08
venue: curated
summary: "GMR 的核心命题很直接：**retargeting matters**。论文指出，humanoid motion tracking policies 依赖人类动作重定向，但人和机器人之间存在 **embodiment gap**。重定向阶段留下的脚滑、不可行姿态、起始姿态不合理等问题，会直接影响后面的 **RL 控制器**。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-03-data-pipeline.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../methods/motion-retargeting-gmr.md
sources:
  - ../../sources/papers/humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# Retargeting Matters

**Retargeting Matters** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 01/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。

> **深读页：** [motion-retargeting-gmr](../methods/motion-retargeting-gmr.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

GMR 的核心命题很直接：**retargeting matters**。论文指出，humanoid motion tracking policies 依赖人类动作重定向，但人和机器人之间存在 **embodiment gap**。重定向阶段留下的脚滑、不可行姿态、起始姿态不合理等问题，会直接影响后面的 **RL 控制器**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GMR | General Motion Retargeting | 把人体/视频动作重定向为机器人可执行参考 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **01 数据 · 重定向 · 遥操作**（#01/42）。
- GMR 的核心命题很直接：**retargeting matters**。论文指出，humanoid motion tracking policies 依赖人类动作重定向，但人和机器人之间存在 **embodiment gap**。重定向阶段留下的脚滑、不可行姿态、起始姿态不合理等问题，会直接影响后面的 **RL 控制器**。
- 这篇论文的价值在于，它没有把 retargeting 当成 **“训练前处理一下数据”** 的小步骤，而是系统评估了 **重定向质量对 motion tracking policy 的影响**。
- 论文比较了 GMR、PHC、ProtoMotions、Unitree 官方重定向等方法，并通过用户研究和 sim2sim 成功率说明：一个重定向结果既要接近源动作，又要适合训练控制器。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 01/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | 斯坦福大学 |
| 出处 | curated |
| 链接 | <https://jaraujo98.github.io/retargeting\_matters/> |

## 核心机制（归纳）

### 1）策展导读要点

GMR 的核心命题很直接：**retargeting matters**。论文指出，humanoid motion tracking policies 依赖人类动作重定向，但人和机器人之间存在 **embodiment gap**。重定向阶段留下的脚滑、不可行姿态、起始姿态不合理等问题，会直接影响后面的 **RL 控制器**。

### 2）策展导读要点

这篇论文的价值在于，它没有把 retargeting 当成 **“训练前处理一下数据”** 的小步骤，而是系统评估了 **重定向质量对 motion tracking policy 的影响**。

### 3）策展导读要点

论文比较了 GMR、PHC、ProtoMotions、Unitree 官方重定向等方法，并通过用户研究和 sim2sim 成功率说明：一个重定向结果既要接近源动作，又要适合训练控制器。

### 4）策展导读要点

如果参考轨迹本身物理上很糟，策略会陷入两难：严格跟踪会摔，保持稳定又会偏离参考。最后得到的控制器可能既不像人，也不够稳。

## 常见误区

1. 重定向/遥操作不是「训练前脚本」——参考质量上限往往 **早于** RL 策略决定。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md](../../sources/papers/humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md)

## 参考来源

- [humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md](../../sources/papers/humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：Retargeting Matters](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_Motion_Retargeting/Retargeting_Matters__General_Motion_Retargeting_for_Humanoid_Motion_Tracking/Retargeting_Matters__General_Motion_Retargeting_for_Humanoid_Motion_Tracking.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
