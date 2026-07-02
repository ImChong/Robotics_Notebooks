---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, loco-manipulation, loco-manip-161-survey, nvidia]
status: complete
updated: 2026-07-02
venue: "project"
code: https://github.com/NVIDIA/Isaac-GR00T
summary: "GR00T N1 是 NVIDIA 开源的人形通才基础模型：把视觉、语言与动作放进同一 VLA 栈，并显式建模状态历史、action chunk、embodiment tag 与跨数据源 post-training；在 RL 身体系统栈与 Loco-Manip 161 篇中均为 VLA/任务接口代表工作。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-09-vla-world-models.md
  - ../entities/gr00t-wholebodycontrol.md
sources:
  - ../../sources/papers/humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/loco_manip_161_survey_148_gr00t-n1.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
---

# GR00T N1

**GR00T N1**（*An Open Foundation Model for Generalist Humanoid Robots*，[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)，[PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)）是 NVIDIA 提出的人形通才基础模型。

> **深读页：** [gr00t-wholebodycontrol](../entities/gr00t-wholebodycontrol.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

GR00T N1 把视觉、语言与动作放进同一个 humanoid foundation model：先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用 ACT/行为克隆、扩散策略/流匹配与 VLA 动作头预测可执行命令；关键不只是「VLA 口号」，而是显式建模状态历史、action chunk、embodiment tag、latent action 与跨数据源 post-training。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#34/42）。
- 在 [人形 Loco-Manip 161 篇技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md) 中属于 **09 人形 VLA、世界模型与通用操作**（#148/161）；本页为合并后的 **单一 canonical 实体**。
- GR00T N1 的目标不是再做一个单项 manipulation policy，而是把视觉、语言和动作放进同一个 humanoid foundation model 里。
- 论文真正有价值的地方在于把机器人动作接口拆得很具体：状态历史、action chunk、embodiment tag、latent action、真实动作标签、合成轨迹和真机 post-training 都要一起进入系统。
- 最底层是大量人类视频和网络视频，中间是神经生成轨迹和仿真轨迹，最上层才是真机机器人数据；真机数据最贵、最贴近身体，但覆盖度不足，人类视频和合成数据覆盖度高，但必须通过 latent action、inverse dynamics 或 post-training 才能落到具体机器人身上。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 原文题目 | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots |
| 机构 | NVIDIA |
| 发表日期 | 2025年3月27日 |
| 项目/代码 | <https://github.com/NVIDIA/Isaac-GR00T> |
| 数据集 | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim> |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 34/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在人形 Loco-Manip 161 篇中

| 字段 | 内容 |
|------|------|
| 编号 | 148/161 |
| 分组 | 09 人形 VLA、世界模型与通用操作 |
| 分类 hub | [loco-manip-161-category-09-vla-world-models](../overview/loco-manip-161-category-09-vla-world-models.md) |
| 索引来源 | [具身智能研究室 · 161 篇人形 Loco-Manip 长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) |

## 核心机制（归纳）

### 1）多模态编码与 VLA 动作头

先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用 ACT/行为克隆模仿学习、扩散策略/流匹配、VLA 多模态动作模型预测可执行动作命令。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

### 2）动作接口与 embodiment 对齐

这听起来像「VLA 控制机器人」，但论文真正有价值的地方恰恰不是这句口号，而是它把机器人动作接口拆得很具体：状态历史、action chunk、embodiment tag、latent action、真实动作标签、合成轨迹和真机 post-training 都要一起进入系统。

### 3）分层数据金字塔

最底层是大量人类视频和网络视频，中间是神经生成轨迹和仿真轨迹，最上层才是真机机器人数据。真机数据最贵、最贴近身体，但覆盖度不足；人类视频和合成数据覆盖度高，但必须通过 latent action、inverse dynamics 或 post-training 才能落到具体机器人身上。

## 常见误区

1. VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。
2. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
3. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 横向评测对照请回到 [Loco-Manip 分类 hub](../overview/loco-manip-161-category-09-vla-world-models.md)、[Loco-Manip 技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md) 与 [42 篇 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)。

## 与其他页面的关系

- 方法深读：[gr00t-wholebodycontrol.md](../entities/gr00t-wholebodycontrol.md)
- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- Loco-Manip 161 篇：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)

## 参考来源

- [humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md](../../sources/papers/humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [loco_manip_161_survey_148_gr00t-n1.md](../../sources/papers/loco_manip_161_survey_148_gr00t-n1.md) — Loco-Manip 161 #148 策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md) — Loco-Manip 161 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md) — Loco-Manip 161 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：GR00T N1 Humanoid Foundation Model](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/GR00T_N1_Humanoid_Foundation_Model/GR00T_N1_Humanoid_Foundation_Model.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [161 篇 Loco-Manip（微信公众号）](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
