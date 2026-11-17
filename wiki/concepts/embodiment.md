---
type: concept
tags: [robotics, cross-embodiment, morphology, policy-interface]
status: complete
updated: 2026-11-17
summary: "机器人本体（embodiment）指约束策略感知与行动的身体、传感器、执行器及控制接口；跨本体数据格式一致不保证动作语义、时序或任务可行性一致。"
related:
  - ./urdf-robot-description.md
  - ./open-x-embodiment.md
  - ./motion-retargeting.md
  - ../entities/inspect-robots.md
  - ../entities/paper-saycan.md
  - ../methods/vla.md
sources:
  - ../../sources/repos/robocurve_inspect_robots_unitree_g1.md
  - ../../sources/papers/saycan_arxiv_2204_01691.md
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
---

# 机器人本体（Embodiment）

## 一句话定义

**机器人本体（embodiment）**是策略实际依附的身体及其感知、动作和控制约束：同一条指令能否执行，取决于机器人能观察什么、控制什么，以及当前身体和状态允许做什么。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| DoF | Degrees of Freedom | 身体可独立运动的自由度 |
| URDF | Unified Robot Description Format | 连杆、关节、惯量及碰撞几何的描述格式 |
| EEF | End-Effector | 末端执行器；动作坐标系和夹爪语义的重要参照 |
| VLA | Vision-Language-Action | 消费观测并产生动作的策略，部署时仍需匹配本体接口 |

## 为什么重要

- **模型输入输出的形状只是一层契约。** 相同维数的动作可能分别表示绝对关节位置、相对增量或末端位姿，不能直接互换。
- **跨本体数据混合保留物理差异。** [Open X-Embodiment](./open-x-embodiment.md) 统一存储 schema 后，连杆尺寸、自由度、动力学、相机视角和控制频率仍需分别处理。
- **任务可行性依赖当前身体和状态。** [SayCan](../entities/paper-saycan.md) 用技能价值函数过滤当前不可行的候选技能，语言上合理并不足以支持实际执行。

## 核心原理：身体如何约束策略接口

| 层次 | 核对内容 | 站内入口 |
|---|---|---|
| 物理身体 | 关节拓扑、连杆尺寸、惯量、限位、末端工具与接触能力 | [URDF](./urdf-robot-description.md) |
| 观测接口 | 相机名与视角、状态键、单位、坐标系和采样时间 | [Inspect Robots](../entities/inspect-robots.md) |
| 动作与控制接口 | 关节或末端动作、绝对或相对、执行频率、夹爪方向和 reset 行为 | [Inspect Robots G1 适配器归档](../../sources/repos/robocurve_inspect_robots_unitree_g1.md) |
| 任务能力 | 当前状态下技能能否成功，是否满足目标任务的身体与环境条件 | [SayCan](../entities/paper-saycan.md) |

```mermaid
flowchart LR
  B["身体与传感执行能力"] --> O["可用观测与动作接口"]
  O --> P["策略或技能选择"]
  B --> F["当前状态下的可行性"]
  P --> F
  F --> E["执行与反馈"]
  E --> O
```

本图归纳上述接口关系，不对应某个官方项目的源码调用流程。

## 工程实践：跨本体迁移前核对什么

1. 固定训练数据、checkpoint 与部署机器人的身份；记录相机、关节顺序、手型及控制版本。
2. 对齐观测、动作的单位与坐标系，以及绝对/相对约定；把动作预测频率、实际执行频率和时延分别记录。
3. 确认所暴露的控制范围。例如 Inspect Robots G1 的双臂适配器不接管腿、腰和平衡，不能据此推断完整步态控制能力。
4. 需要改变动作空间时，查看[运动重定向](./motion-retargeting.md)，并重新验证目标本体的限位、接触与任务成功条件。

## 局限与术语边界

- URDF 承载身体模型的一部分，不能独自表达全部相机同步、策略动作语义和运行时控制契约。
- 数据集里的 `embodiment` 字段、框架中的 `Embodiment` 类，以及 SayCan 的 Embodiment 测试类别各有具体定义，阅读时应回到对应数据说明、API 或论文。
- 本页基于知识库既有来源归档归纳概念；没有新增原论文通读、源码运行或真机迁移成功的声明。

## 关联页面

- [URDF](./urdf-robot-description.md) — 身体结构与惯量的建模入口
- [Open X-Embodiment](./open-x-embodiment.md) — 跨本体数据混合与格式统一
- [运动重定向](./motion-retargeting.md) — 目标身体约束下的运动映射
- [Inspect Robots](../entities/inspect-robots.md) — 策略与身体接口的评测契约
- [SayCan](../entities/paper-saycan.md) — 语言任务与物理可行性的分解
- [VLA](../methods/vla.md) — 观测到动作的策略接口

## 参考来源

- [Inspect Robots G1 适配器归档](../../sources/repos/robocurve_inspect_robots_unitree_g1.md) — 观测、16 维绝对关节动作、频率和身体控制范围
- [SayCan 论文核对归档](../../sources/papers/saycan_arxiv_2204_01691.md) — 技能词表、物理 affordance 与 Embodiment 任务类别
- [机器人学习三阶段一手资料索引](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md) — OXE/RT-X 的跨本体数据与策略边界

## 推荐继续阅读

- [Inspect Robots Concepts](https://docs.inspectrobots.org/guide/concepts/)
- [Open X-Embodiment 项目页](https://robotics-transformer-x.github.io/)
