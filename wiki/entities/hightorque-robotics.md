---
type: entity
tags: [humanoid, hardware, open-source, hightorque, robotics, research, manipulation]
status: complete
updated: 2026-07-09
related:
  - ./humanoid-robot.md
  - ./open-source-humanoid-hardware.md
  - ./unitree.md
  - ./paper-host-humanoid-standingup.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ./lerobot.md
sources:
  - ../../sources/repos/hightorque_robotics.md
summary: "高擎机电（HighTorque Robotics）提供桌面级 Mini Pi Plus 人形与 Panthera-HT 开源六轴臂；GitHub 组织覆盖 Isaac Lab/Gym 全身跟踪、PPO 行走、OCS2 MPC+WBC 与 ROS sim2real 全链路。"
---

# 高擎机电（HighTorque Robotics）

## 一句话定义

**高擎机电（HighTorque Robotics）** 是广州高擎机电科技有限公司旗下品牌，定位「具身智能时代的 PC」：用自研高功率密度关节模组，把 **65 cm 级小型人形（Mini Pi Plus）** 与 **开源六轴臂（Panthera-HT）** 做成科研、教学与竞赛（含 RoboCup 人形小尺寸）可复用的二次开发平台；官方 GitHub 组织 **[HighTorque-Robotics](https://github.com/HighTorque-Robotics)** 提供从仿真训练到真机部署的完整开源管线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | Mini Pi Plus 约 26–27 关节自由度 |
| RL | Reinforcement Learning | PPO 行走与全身跟踪训练主线 |
| Sim2Real | Simulation to Real | 仿真策略经 ONNX/JIT 迁移真机 |
| WBC | Whole-Body Control | hi_dynamic_control 中的全身协调控制层 |
| MPC | Model Predictive Control | OCS2 非线性模型预测控制 |
| Isaac Lab | NVIDIA Isaac Lab | BeyondMimic / Teacher–Student 训练环境 |
| Isaac Gym | NVIDIA Isaac Gym | livelybot_pi_rl_baseline 经典并行仿真 |
| GMR | General Motion Retargeting | 人体/动捕数据重定向到 Pi Plus 骨架 |
| ROS | Robot Operating System | sim2real / hi_dynamic_control 真机中间件 |
| BOM | Bill of Materials | Panthera-HT 结构开源物料清单 |

## 为什么重要

- **桌面级人形硬件入口：** 与 Unitree G1/H1 等全尺寸平台互补，Mini Pi Plus 以更小形态、更低门槛覆盖 **RL / 模仿学习 / Sim2Real** 与 **RoboCup 竞技** 场景。
- **开源管线完整：** 同一组织内并存 **Isaac Lab 全身跟踪（BeyondMimic）**、**Isaac Gym PPO 基线**、**OCS2 MPC+WBC** 与 **ROS sim2real**，便于按方法选型而非只买硬件。
- **机械臂 + 人形双产品线：** Panthera-HT 把 **SDK / ROS2 / LeRobot** 接到同一关节模组生态，适合 loco-manip 与具身数据采集实验。
- **社区与赛事背书：** ICRA 2026 全球发布 Mini Pi Plus；Panthera-HT 为 WBCD Challenge 官方平台；HoST 起身控制已扩展 Mini Pi 支持。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 高擎机电（HighTorque Robotics） |
| 成立 | 2020（广州） |
| GitHub | <https://github.com/HighTorque-Robotics>（36 公开仓，2026-07） |
| 官网 | <https://hightorque.cn/> |
| 旗舰人形 | **Mini Pi Plus**（~65 cm，26–27 DoF，自研集成伺服关节） |
| 旗舰机械臂 | **Panthera-HT**（开源六轴，教学/创客/具身数据采集） |

## 产品线概览

### Mini Pi / Mini Pi Plus（小型人形）

面向科研、教育与 **RoboCup Humanoid League Small Size** 的集成小型人形。硬件强调高集成度与实时控制；软件侧官方维护多条并行训练/部署路径：

| 能力方向 | 代表仓库 | 技术要点 |
|----------|----------|----------|
| 全身运动跟踪 | [Mini-Pi-Plus_BeyondMimic](https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic) | Isaac Sim 5 + Isaac Lab 2.2；GMR retarget；**零调参** sim-to-real 跟踪 |
| RL 行走基线 | [livelybot_pi_rl_baseline](https://github.com/HighTorque-Robotics/livelybot_pi_rl_baseline) | Isaac Gym + PPO；含 Isaac Gym → MuJoCo **sim2sim** |
| Teacher–Student | [Pi_Isaaclab](https://github.com/HighTorque-Robotics/Pi_Isaaclab) | Isaac Lab 蒸馏训练框架 |
| 模型基控制 | [hi_dynamic_control](https://github.com/HighTorque-Robotics/hi_dynamic_control) | **OCS2 NMPC + WBC**；串并联踝 Hi 平台；Gazebo/真机 |
| 真机部署 | [sim2real](https://github.com/HighTorque-Robotics/sim2real) | ROS Noetic 部署栈 |
| 竞技集成 | [RoboCup_Workspace](https://github.com/HighTorque-Robotics/RoboCup_Workspace) | 行为/规划/视觉/网络全栈 |
| 起身扩展 | [HoST](https://github.com/HighTorque-Robotics/HoST) | RSS 2025 起身控制在 Mini Pi 上的 fork |

### Panthera-HT（开源六轴机械臂）

源自社区 [Ragtime_Panthera](https://github.com/Ragtime-LAB/Ragtime_Panthera)，与高擎联合落地为完整创客产品：**结构全开源**（钣金 + 3D 打印 + 自研行星关节模组），控制覆盖位置/速度/力矩/阻抗、重力与摩擦补偿、双臂主从遥操与拖动示教；并接入 **LeRobot** 模仿学习管线。主索引：[Panthera-HT_Main](https://github.com/HighTorque-Robotics/Panthera-HT_Main) / [Hub](https://hightorque.cn/Panthera-HT_Hub/)。

## 流程总览（Mini Pi Plus 全身跟踪 → 真机）

```mermaid
flowchart LR
  A[动捕 / BVH / LAFAN1] --> B[GMR Retarget]
  B --> C[CSV / NPZ 运动库]
  C --> D[Isaac Lab + BeyondMimic<br/>rsl_rl PPO 训练]
  D --> E[ONNX / JIT 导出]
  E --> F[MuJoCo sim2sim 验证]
  F --> G[ROS sim2real / 真机部署]
```

经典 **Isaac Gym 行走** 路径可简化为：`PPO 训练 → play 导出 JIT → sim2sim.py（MuJoCo）→ sim2real launch`。

## 与同类平台的关系

| 对比维度 | HighTorque Mini Pi Plus | Unitree G1 等 |
|----------|-------------------------|---------------|
| 形态 | ~65 cm 桌面级小型人形 | 全尺寸/教育级人形 |
| 开源深度 | 组织内多仓覆盖训练到部署 | 硬件成熟 + 社区 legged_gym 生态极强 |
| 典型场景 | RoboCup 小尺寸、课堂/桌面实验 | 通用 loco-manip 与大规模 RL 复现 |
| 机械臂线 | Panthera-HT 同源关节模组 | 以人形/四足为主 |

二者在 sim2real 语境下常并列出现（如 [HoST](./paper-host-humanoid-standingup.md) 同时支持 G1 与 Mini Pi）；选型时可与 [开源人形硬件方案对比](./open-source-humanoid-hardware.md) 及 [Unitree](./unitree.md) 交叉阅读。

## 常见误区

- **不是玩具级舵机方案：** 官方强调自研高功率密度关节与工业级控制栈，与廉价舵机双足不同。
- **不止一条软件栈：** BeyondMimic（跟踪）、PPO 基线（行走）、OCS2（MPC+WBC）环境依赖不同，需按任务选型而非混装。
- **GitHub 组织 ≠ 全部历史仓：** 部分早期 RL 基线曾以 `HighTorque-Locomotion` 等账号发布，当前维护以 `HighTorque-Robotics` 组织为准。

## 推荐继续阅读

- 官方 GitHub 组织：<https://github.com/HighTorque-Robotics>
- Mini Pi Plus 产品页：<https://store.hightorque.cn/products/bipedal-robot-mini-pi-plus>
- Panthera-HT Hub：<https://hightorque.cn/Panthera-HT_Hub/>
- BeyondMimic 上游：<https://github.com/HybridRobotics/whole_body_tracking>
- HoST 论文实现（主仓）：<https://github.com/InternRobotics/HoST>

## 参考来源

- [HighTorque Robotics GitHub 组织归档](../../sources/repos/hightorque_robotics.md)
- 官网：<https://hightorque.cn/en/about-us>
- [HoST 官方仓库归档](../../sources/repos/host_internrobotics.md)（Mini Pi 扩展提及）

## 关联页面

- [人形机器人](./humanoid-robot.md)
- [开源人形硬件方案对比](./open-source-humanoid-hardware.md)
- [Unitree（宇树科技）](./unitree.md)
- [HoST：人形起身控制](./paper-host-humanoid-standingup.md)
- [Sim2Real](../concepts/sim2real.md)
- [强化学习](../methods/reinforcement-learning.md)
- [LeRobot](./lerobot.md)

## 一句话记忆

> HighTorque 把「小型人形 + 开源机械臂」做成带完整仿真训练与真机部署管线的桌面级具身平台，是 Unitree 全尺寸生态之外 RoboCup 与教学场景的重要国产硬件入口。
