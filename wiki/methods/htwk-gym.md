---
type: method
tags: [rl, isaac-gym, booster-t1, soccer, locomotion]
status: drafting
updated: 2026-04-27
related:
  - ../tasks/humanoid-soccer.md
  - ../entities/booster-robocup-demo.md
  - ./reinforcement-learning.md
  - ../entities/isaac-gym-isaac-lab.md
sources:
  - ../../sources/repos/htwk_gym.md
summary: "htwk-gym 是由 HTWK Leipzig 开发的人形机器人足球 RL 框架，专注于 Booster T1/K1 平台的参数化步行与踢球技能训练。"
---

# htwk-gym

**htwk-gym** 是一个开源的强化学习（RL）框架，专门针对人形机器人足球（Humanoid Soccer）竞赛设计。该框架由 RoboCup 强队 HTWK Leipzig 维护，在 **Booster T1/K1** 平台上经过了广泛的验证。

## 核心组件

### 1. 参数化步行 (Parameterized Walking)
htwk-gym 并不训练单一的行走策略，而是训练一个**参数化控制器**。
- **控制输入**：步频、脚部偏航角、身体姿态参数。
- **优势**：允许上层决策系统（如 ROS 2 状态机）根据球场局势动态调整行走风格，实现灵巧的变向和截球。

### 2. 足球专项任务环境
- **`T1/Kicking`**：包含球感知奖励、碰撞惩罚以及目标速度引导，用于训练精准的踢球动作。
- **`T1/Walking`**：侧重于高速平滑移动。

### 3. 多端导出与部署 (Deployment Pipeline)
- **PyTorch JIT**：用于仿真评估和高性能 PC 部署。
- **TFLite 量化**：通过 TensorFlow Lite 将策略量化，使其能够运行在 Booster 机器人内部的嵌入式 ARM/NVIDIA Orin 模块上。

## 主要技术路线

| 模块 | 实现方案 | 优势 |
|------|---------|------|
| **运动控制** | 参数化步态策略 (Parameterized Gait) | 允许上层逻辑动态调节速度与步频 |
| **训练环境** | Isaac Gym 并行仿真 | 极高的样本采集效率，缩短训练时间 |
| **Sim-to-Real** | 领域随机化 + 跨仿真验证 (MuJoCo) | 提高策略在真实不平整草地上的鲁棒性 |
| **部署优化** | TFLite 量化导出 | 适配机器人端侧 ARM/Orin 算力平台 |

## 技术细节

- **基础引擎**：使用 NVIDIA **Isaac Gym** 实现大规模并行训练，可在数分钟内完成基础步态学习。
- **跨仿真验证**：支持将训练好的模型在 **MuJoCo** 环境中进行独立验证，以减少 Sim-to-Real 的鸿沟。
- **Streamlit 调试工具**：提供了一个名为 `obs_editor` 的 Web 界面，开发者可以实时拉动滑块调整训练参数或观察策略表现。

## 在项目中的角色

htwk-gym 作为底层“技能工厂”，为 [Booster RoboCup Demo](../entities/booster-robocup-demo.md) 提供高性能的原子技能（如 `RLVisionKick`）。

## 参考来源

- [NaoHTWK/htwk-gym 源码仓库](../../sources/repos/htwk_gym.md)
- HTWK Leipzig RoboCup Team Documentation

## 关联页面

- [Humanoid Soccer](../tasks/humanoid-soccer.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [Sim2Real](../concepts/sim2real.md)
- [Domain Randomization](../concepts/domain-randomization.md)
- [Booster Robotics RoboCup Demo](../entities/booster-robocup-demo.md)
- [Isaac Gym / Isaac Lab](../entities/isaac-gym-isaac-lab.md)

