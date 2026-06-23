---
type: entity
tags: [humanoid, hardware, open-source, berkeley, reinforcement-learning, qdd]
status: complete
updated: 2026-05-18
related:
  - ./humanoid-robot.md
  - ./open-source-humanoid-hardware.md
  - ../overview/robot-open-source-wechat-issue01-curator.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md
summary: "Berkeley Humanoid Lite（BHL）：UC Berkeley Hybrid Robotics 的低成本准直驱人形开源项目，含 MuJoCo 仿真、PPO 与遥操作等入门链路。"
institutions: [berkeley]

---

# Berkeley Humanoid Lite（BHL）

## 一句话定义

**Berkeley Humanoid Lite** 是 **UC Berkeley Hybrid Robotics** 的 **轻量人形** 开源方案：门户 **[lite.berkeley-humanoid.org](https://lite.berkeley-humanoid.org/)**，源码与仿真/学习脚本在 **[Berkeley-Humanoid-Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- **低成本 QDD 叙事**：在 [开源人形硬件对比](./open-source-humanoid-hardware.md) 中常与商业整机对照，作为 **动力学透明 + RL 友好** 的参考轴。
- **课程式链路**：文中列出 **MuJoCo**、**PPO** 与 **遥操作 IK**，适合作为 **Sim2Real** 实验前的 README 级索引。

## 开源入口（策展摘录）

| 类型 | 链接 |
|------|------|
| 项目门户 | [lite.berkeley-humanoid.org](https://lite.berkeley-humanoid.org/) |
| GitBook 文档（转载） | [berkeley-humanoid-lite.gitbook.io/docs](https://berkeley-humanoid-lite.gitbook.io/docs) |
| 主仓库 | [HybridRobotics/Berkeley-Humanoid-Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite) |

## 关联页面

- [人形机器人](./humanoid-robot.md)
- [开源人形硬件方案对比](./open-source-humanoid-hardware.md)
- [强化学习](../methods/reinforcement-learning.md)
- [机器人开源宝库（微信策展第01期）索引](../overview/robot-open-source-wechat-issue01-curator.md)

## 推荐继续阅读

- [开源人形硬件方案对比 · Berkeley Humanoid 段](./open-source-humanoid-hardware.md#1-berkeley-humanoid-准直接驱动派)

## 参考来源

- [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)
