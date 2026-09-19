---
type: concept
tags: [data-collection, supervision, behavior-cloning, rollout, intervention, ego, reward-learning]
status: complete
updated: 2026-09-19
related:
  - ./embodied-data-flywheel-minimal-closed-loop.md
  - ./embodied-data-collection-four-layers-taxonomy.md
  - ../methods/behavior-cloning.md
  - ../methods/imitation-learning.md
  - ../methods/inverse-reinforcement-learning.md
  - ../overview/embodied-data-collection-to-flywheel-album.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/blogs/wechat_jushen_qianyan_robot_data_supervision_types_2026-09-14.md
summary: "具身数据不是同质大池：成功示范、rollout+结果、人工接管、Ego RGB/Ego+轨迹、偏好比较分别回答动作/好坏/恢复/表征/排序等不同训练问题。"
---

# 机器人数据：监督信号类型分流

> 知识编译自 [具身智能前沿 · 示范、失败和接管分别教什么（2026-09-14）](https://mp.weixin.qq.com/s?__biz=Mzg5OTY3ODkzNg==&mid=2247494519&idx=1&sn=7970dffc18701098a5fbd545d328649c)；本页保留 **标签语义**，避免混池训练。

## 一句话定义

**同一次任务** 可能产生相机、关节、动作、成败、接管与 Ego 视频等多种记录，但它们回答的 **训练提问** 不同；混成「具身数据大池」会丢失监督含义，让模型收到比任务所需更粗或更错的答案。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BC | Behavior Cloning | 给定观测回归示范动作 |
| IL | Imitation Learning | 从演示学习策略的总称 |
| Ego | Egocentric | 第一人称视频/轨迹 |
| VLA | Vision-Language-Action | 可接收多源信号的后训练对象 |
| RECAP | — | 同时使用示范、rollout、纠正的迭代框架 |
| RL | Reinforcement Learning | 可用 rollout+结果/奖励改进策略 |

## 为什么重要

- **BC 要「当时该怎样动」**；纯 RGB Ego **无法** 给出下一帧控制指令。
- **Rollout 失败** 说明「这段不好」，**不自动** 提供逐帧正确动作。
- **接管** 提供 **偏离状态附近** 的恢复动作，与完整示范粒度不同。
- **飞轮有效** 的条件（见 [最小闭环](./embodied-data-flywheel-minimal-closed-loop.md)）在训练层进一步具体为：**每种记录以其真实监督形式改变对的系统部分**。

## 最小分流表

| 数据或反馈 | 主要回答 | 常见训练职责 | 进入训练前必要条件 |
|------------|----------|--------------|-------------------|
| **成功示范** | 此状态下专家如何完成？ | BC / 动作监督 | obs–action 时序对齐；动作可执行或可重定向 |
| **Rollout + 结果** | 当前策略这次做得怎样？ | 价值/奖励/筛选/RL | 任务口径、成败或奖励、策略与场景版本 |
| **人工接管/纠正** | 偏离后如何拉回？ | 恢复、困难状态改进 | 接管时机、原策略上下文、纠正动作与后续结果 |
| **Ego RGB  only** | 人在做什么、步骤语义？ | 视觉表征、任务先验 | 不伪装成动作标签 |
| **Ego + 手部 3D 轨迹** | 手在相机系如何动？ | 对齐后 IL 或与 robot demo 共训 | 坐标/动作空间对齐、可行性检查 |
| **轨迹偏好/比较** | 两路径哪个更好？ | 奖励模型、偏好优化 | 比较口径与上下文一致 |

> 同一轮训练可 **同时使用** 多类信号（如 RECAP）；关键是 **不假装每行提供相同答案**。

## 分类型要点

### 成功示范（含 teleop）

- 定义：**观测—动作配对** 轨迹，非「有人做过」的视频。
- 优势：细粒度动作（方向、闭合时机、末端位姿）。
- 边界：覆盖专家状态；部署 **covariate shift** 后需结果与纠正（见 [BC Mysteries](./behavioral-cloning-mysteries.md)）。

### Ego 分层

| 层级 | 内容 | 训练用途 |
|------|------|----------|
| Ego RGB | 连续画面 | R3M 类表征预训练 |
| Ego + 3D 手/设备 | Project Aria 等 | EgoMimic 类对齐后动作学习 |
| 已对齐 robot 接口 | 经 IK/重定向 | 与 teleop demo 同等动作监督 |

字段应区分三者，勿统称「人类视频示范」。

### Rollout 与接管

- **Rollout：** 策略访问 **专家未走过** 的状态；失败只标「未达目标」，**非** 逐帧专家动作。
- **接管：** 在偏离点记录 **比继续执行更好** 的纠正；需保存触发前状态与纠正后结果。
- **RECAP：** rollout 结果与纠正 **分工**，非「失败轨迹全部当示范模仿」。

### 结果与偏好

- **成功/失败：** 裁判信号；稀疏时需子任务、失败位置、介入与场景才能归因。
- **偏好（Mehta & Losey 等）：** 完整 demo / 局部纠正 / 两轨迹比较 **粒度不同**——缺动作时补偏好，缺排序时补示范或纠正。

## 采集元数据清单

部署链上至少拆分保存：

1. **动作可用性** — 本体控制 / 可重定向 EE / 无动作
2. **结果口径** — 成功、失败、部分完成的判定规则
3. **人工介入** — 何时、谁、接管前策略输出、纠正后是否恢复
4. **上下文与版本** — 场景、物体、传感器、策略、控制器版本
5. **训练去向** — 动作监督 / 表征预训练 / 价值 / 恢复 / 仅回归测试

## 局限与风险

- 表为 **最小分流**；具体算法（RECAP、ConRFT、TOP-AWR 等）组合方式见各论文实体。
- 偏好不能替代缺失的动作标签；成功标签不能替代局部恢复动作。
- 与 [HumanNet 对比表](../comparisons/humannet-table1-human-video-corpora.md) 的 Direct/Indirect 划分一致：Indirect 数据勿当 Direct 动作监督。

## 关联页面

- [飞轮最小闭环](./embodied-data-flywheel-minimal-closed-loop.md)
- [四层采集术语](./embodied-data-collection-four-layers-taxonomy.md)
- [Behavior Cloning](../methods/behavior-cloning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [系列专辑](../overview/embodied-data-collection-to-flywheel-album.md)

## 参考来源

- [wechat_jushen_qianyan_robot_data_supervision_types_2026-09-14.md](../../sources/blogs/wechat_jushen_qianyan_robot_data_supervision_types_2026-09-14.md)
- [机器人数据不是一个池子（微信公众号）](https://mp.weixin.qq.com/s?__biz=Mzg5OTY3ODkzNg==&mid=2247494519&idx=1&sn=7970dffc18701098a5fbd545d328649c)

## 推荐继续阅读

- Mehta & Losey, *Unified Learning from Demonstrations, Corrections, and Preferences* — [arXiv:2207.03395](https://arxiv.org/abs/2207.03395)
- EgoMimic — [arXiv:2410.24221](https://arxiv.org/abs/2410.24221)
- Nair et al., R3M — [arXiv:2203.12601](https://arxiv.org/abs/2203.12601)
