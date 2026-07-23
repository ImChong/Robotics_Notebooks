---
type: overview
tags: [humanoid, history, platform, education]
status: complete
updated: 2026-07-23
related:
  - ../entities/humanoid-robot.md
  - ../entities/unitree-g1.md
  - ../entities/boston-dynamics.md
  - ../entities/openloong.md
  - ../entities/x-humanoid.md
  - ../concepts/lip-zmp.md
  - ./humanoid-algorithm-research-status.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
  - ../../sources/papers/humanoid_hardware.md
summary: "人形机器人发展简史：从早期双足样机、ASIMO/HRP 准静态时代，到 Atlas 高动态与 RL 量产科研平台（G1 等）；课程 1.1，标出控制范式转折。"
---

# 人形机器人发展历史

## 一句话定义

**人形机器人发展历史**梳理双足类人平台从实验室样机到可量产科研整机的关键里程碑，帮助理解今日 [G1](../entities/unitree-g1.md) 等课程平台为何同时继承 **模型基平衡直觉** 与 **RL 全身控制** 两套遗产——课程第 1.1 节。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ZMP | Zero Moment Point | 准静态行走主流判据 |
| HRP | Humanoid Robotics Project | 日本大型人形项目系列 |
| LIP | Linear Inverted Pendulum | 与 ZMP 搭配的简化模型 |
| RL | Reinforcement Learning | 近十年动态技能主范式 |
| WBC | Whole-Body Control | 多任务全身 QP 控制 |
| DoF | Degrees of Freedom | 关节自由度 |

## 为什么重要

- 否则易把 **ASIMO 时代的准静态行走** 与 **Atlas/G1 的高动态 RL** 混为一谈，导致课程后面「为什么直接上 PPO」失去语境。
- 硬件成本与开源训练栈的变化，解释了为何系统课能以单台 G1 + 仿真覆盖行走→导航→足球→大模型整条链。
- 读史是为了 **选型与预期管理**，不是背年份。

## 核心阶段

| 阶段 | 时代特征 | 代表 | 控制特征 |
|------|----------|------|----------|
| 早期探索 | 实验室证明「能走」 | WABOT、WL 等 | 开环/简单反馈，演示为主 |
| 准静态成熟 | 可靠性与公众认知 | Honda ASIMO、HRP 系列 | [ZMP/LIP](../concepts/lip-zmp.md)、预观控制 |
| 高动态力控 | 跑跳翻转进大众视野 | Boston Dynamics Atlas 等 | 模型预测、液压/力控 |
| 学习革命 | 仿真并行 + Sim2Real | Cassie/Digit、各家 RL 人形 | PPO/模仿、域随机 |
| 量产科研平台 | 高校可负担整机 | Unitree G1/H1、OpenLoong、天工等 | 中低成本 + 开源栈 |

```mermaid
flowchart LR
  E1["1970s–90s<br/>早期双足样机"] --> E2["2000s<br/>ZMP 准静态成熟"]
  E2 --> E3["2010s<br/>高动态力控演示"]
  E3 --> E4["2020s<br/>RL 与量产科研平台"]
```

## 工程实践（如何用这份历史）

1. **对照本库实体**：形态定义见 [Humanoid Robot](../entities/humanoid-robot.md)；当代平台见 [G1](../entities/unitree-g1.md)、[OpenLoong](../entities/openloong.md)、[X-Humanoid](../entities/x-humanoid.md)、[Boston Dynamics](../entities/boston-dynamics.md)。
2. **课程作业落点**：仿真运动控制落在 **学习期之后** 的工具链，不必复刻 ASIMO 全套模型基栈，但应理解 ZMP 仍出现在许多混合架构与面试问题中。
3. **读论文时标时代**：2015 前「行走」论文与 2023 后「运动跟踪」论文的默认假设完全不同。

## 与算法现状的分工

| 本页 | [算法研究现状](./humanoid-algorithm-research-status.md) |
|------|----------------------------------------------------------|
| 时间线与平台 | 分层方法地图 |
| 控制范式转折 | 运动/导航/足球/大模型现状 |

## 局限与风险

- 公开叙事省略失败项目与非公开分支，不是完整产业史。
- 「世界第一」类营销用语不可当作技术史结论。
- 液压传奇平台的维护/成本曲线与电驱量产平台不可直接类比。

## 关联页面

- [人形算法研究现状](./humanoid-algorithm-research-status.md)
- [LIP / ZMP](../concepts/lip-zmp.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)
- [Humanoid Robot](../entities/humanoid-robot.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)
- [humanoid_hardware 资料](../../sources/papers/humanoid_hardware.md)

## 推荐继续阅读

- Boston Dynamics 技术叙事与 Atlas 公开演示归档
- Honda ASIMO / HRP 历史综述类公开材料
