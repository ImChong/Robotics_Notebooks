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
  - ./humanoid-algorithm-research-status.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
  - ../../sources/papers/humanoid_hardware.md
summary: "人形机器人发展简史：从早期双足样机与 Honda ASIMO，到 Atlas/Digit 动态控制与近年量产教育平台（G1 等），标出控制范式从模型基到学习基的转折。"
---

# 人形机器人发展历史

## 一句话定义

**人形机器人发展历史**梳理双足类人平台从实验室样机到可量产科研整机的关键里程碑，帮助理解今日 [G1](../entities/unitree-g1.md) 等课程平台为何同时继承 **模型基平衡直觉** 与 **RL 全身控制** 两套遗产。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ZMP | Zero Moment Point | 早期稳定行走的主流判据 |
| HRP | Humanoid Robotics Project | 日本通产省人形项目系列 |
| RL | Reinforcement Learning | 近十年动态技能主训练范式 |
| WBC | Whole-Body Control | 多任务全身二次规划控制 |
| DoF | Degrees of Freedom | 关节自由度，衡量形态复杂度 |

## 为什么重要

- 课程第 1.1 节的「现状」必须以时间线为锚，否则易把 **ASIMO 时代的准静态行走** 与 **Atlas/G1 的高动态 RL** 混为一谈。
- 硬件成本与开源栈的变化解释了为何教学课能以 G1 + 仿真完成整条系统链。

## 核心阶段（简表）

| 阶段 | 代表 | 控制特征 |
|------|------|----------|
| 早期双足 | WABOT、WL 系列等 | 开环/简单反馈，演示为主 |
| 准静态成熟期 | Honda ASIMO、HRP | ZMP / 预观控制，可靠性优先 |
| 高动态液压/力控 | Boston Dynamics Atlas 等 | 模型预测 + 力控，跑跳翻转 |
| 学习革命 | Cassie/Digit、多家人形 RL | PPO/模仿 + Sim2Real |
| 量产科研平台 | Unitree G1/H1、OpenLoong 等 | 中低成本 + 开源训练栈 |

## 工程实践

- 读史时对照本库节点：形态与选型见 [Humanoid Robot](../entities/humanoid-robot.md)；当代平台见 [G1](../entities/unitree-g1.md)、[OpenLoong](../entities/openloong.md)。
- 课程作业「仿真运动控制」落在 **学习期之后** 的工具链上，不必复刻 ASIMO 的全部模型基栈。

## 局限与风险

- 公开宣传时间线常省略失败项目与军工分支，不宜当作完整产业史。
- 「历史」节点不替代 [算法研究现状](./humanoid-algorithm-research-status.md)。

## 关联页面

- [人形算法研究现状](./humanoid-algorithm-research-status.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)
- [Humanoid Robot](../entities/humanoid-robot.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)
- [humanoid_hardware 资料](../../sources/papers/humanoid_hardware.md)

## 推荐继续阅读

- Boston Dynamics 技术叙事与 Atlas 公开演示时间线（公司博客 / 视频归档）
