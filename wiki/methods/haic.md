---
type: method
tags: [rl, world-model, teacher-student, imitation-learning, sim2real, paper, humanoid, motion-control, body-system-stack, hkust-gz, xiaomi-robotics, eth, hkust, tsinghua]
status: complete
updated: 2026-07-22
venue: curated
related:
  - ../entities/paper-haic.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/loco-manip-contact-category-05-vla-world-models.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ./model-based-rl.md
  - ./amp-reward.md
  - ../concepts/privileged-training.md
sources:
  - ../../sources/papers/motion_control_projects.md
  - ../../sources/papers/humanoid_rl_stack_38_haic_humanoid_agile_object_interaction_control_v.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
summary: "HAIC（Humanoid Agile Object Interaction Control）用 dynamics-aware world model 从本体历史预测欠驱动物体的速度/加速度，并通过非对称微调把特权动力学信息蒸馏到可部署学生策略。"
---

# HAIC: 动力学感知世界模型控制

**HAIC**（*Humanoid Agile Object Interaction Control via Dynamics-Aware World Model*）不是泛称的分层控制器，而是面向 **underactuated objects** 的人形控制方法：从可部署的本体历史预测对象高阶动力学状态，并把预测投影到几何先验，形成动态占据与接触可供性。

## 两阶段训练范式

HAIC 的核心在于如何平滑地从“全知全能”的教师过渡到“仅靠观测”的学生。

### 第一阶段：联合预训练 (Joint Pre-training)
- **教师策略 (Teacher Policy)**：输入特权对象状态、几何和接触信息，输出专家动作和价值估计。
- **动力学世界模型 (Dynamics-Aware World Model)**：输入本体历史，预测对象 velocity / acceleration 等高阶状态。
- **学生策略 (Student Policy)**：输入本体历史、世界模型预测和动态占据表示，学习可部署动作。
- **目标**：让学生在训练早期获得接近 teacher 的物理后果感知，而不是只记忆接触姿态。

### 第二阶段：非对称微调 (Asymmetric Fine-tuning)
- **教师/critic 保持特权视角**：训练侧仍能访问更完整的对象状态。
- **世界模型跟随 student 分布**：持续适配学生策略探索出的状态分布，减少 rollout 分布偏移。
- **学生策略部署约束**：部署时不依赖外部对象状态估计，只用本体历史和 learned world model。
- **结果**：在滑板、推车、拉车和箱子跨地形任务中补偿欠驱动物体的惯性扰动与视觉遮挡。

## 主要技术路线

| 模块 | 角色 | 训练方法 |
|------|-----|---------|
| **教师策略** | 特权动力学专家 | 使用对象状态、几何和接触信息训练 |
| **世界模型** | 可部署动力学感知层 | 从 proprioceptive history 预测 velocity / acceleration |
| **动态占据** | 空间接地表示 | 把预测投影到几何先验，给出碰撞边界与接触可供性 |
| **学生策略** | 真机部署策略 | 只依赖本体历史和 world-model 输出 |
| **训练范式** | 非对称微调 | 让 world model 适配 student rollout 分布 |

## 技术特色

- **预测动力学而非图像**：HAIC 关心对象速度、加速度和动态占据，而不是生成未来帧。
- **本体历史补盲区**：对象遮挡相机或遮住地面时，策略仍可从接触反馈和身体运动推断后果。
- **开源路径清晰**：官方仓库含 `scripts/train.py`、`scripts/play.py`、Isaac Sim 任务配置与 MuJoCo Sim2Sim 入口。

## 典型应用场景

- **滑板 / 推车 / 拉车**：对象有非完整约束和独立惯性，不能当作刚性末端目标。
- **装箱后运输**：多对象顺序交互要求策略记住已改变的载荷与接触状态。
- **坡面 / 楼梯 / 平台复合地形**：箱体或车体遮挡视觉时，动态占据可辅助避碰和稳定。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HAIC | Humanoid Agile Object Interaction Control | 面向欠驱动物体的人形交互控制框架 |
| WM | World Model | 从本体历史预测对象高阶动力学状态 |
| HOI | Humanoid-Object Interaction | 人形机器人与物体的全身交互任务 |
| Sim2Sim | Simulation to Simulation | Isaac Sim 策略到 MuJoCo 验证/部署链路 |
| RL | Reinforcement Learning | 通过交互优化长期回报的训练范式 |

## Survey 坐标（策展索引）

### 在 具身智能研究室 · 42 篇 humanoid RL 运动控制长文 中

| 字段 | 内容 |
|------|------|
| 编号 | 38/42 |
| 系统栈层 | 05 接触 · 柔顺 · 安全恢复 |
| 出处 | curated |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

## 论文实体

- [HAIC 论文实体](../entities/paper-haic.md) — 论文级页面，包含项目页、开源状态、关键实验与源码运行时序图。

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- [humanoid_rl_stack_38_haic_humanoid_agile_object_interaction_control_v.md](../../sources/papers/humanoid_rl_stack_38_haic_humanoid_agile_object_interaction_control_v.md)
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)
- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
- 官方项目页：<https://haic-humanoid.github.io/>
- 官方代码：<https://github.com/ldt29/HAIC>

## 关联页面

- [Model-Based RL](./model-based-rl.md) — 世界模型在 HAIC 中作为特征提取器。
- [Privileged Training (特权信息训练)](../concepts/privileged-training.md)
- [Sim2Real](../concepts/sim2real.md) — 教师-学生蒸馏是解决 Sim2Real 差距的标准模式。
- [Loco-Manip 接触分类 05：VLA 与世界模型调用](../overview/loco-manip-contact-category-05-vla-world-models.md)
