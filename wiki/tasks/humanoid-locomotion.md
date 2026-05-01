---
type: task
tags: [humanoid, locomotion, whole-body-control]
status: complete
summary: "人形机器人在复杂地形下的平衡与移动任务，强调高维动力学处理、环境感知以及全身肢体协调。"
---

# Humanoid Locomotion (人形机器人移动)

**Humanoid Locomotion**：使双足类人机器人能够在复杂、非结构化的地形中，保持平衡的同时实现高效、鲁棒的位移，并具备全身协调（Whole-body Coordination）能力。

## 一句话定义

让两条腿（甚至加上手和膝盖）在各种烂路上走稳、走远、走得像人。

## 核心挑战

1. **高维非线性动力学**：人形机器人具有数十个自由度，其动力学模型高度复杂且存在欠驱动（Under-actuated）阶段。
2. **接触力学建模**：涉及足端、手部或膝盖与地形的断续接触，传统的基于模型的控制（如 MPC）在处理多点接触时计算量巨大。
3. **环境感知与反应**：需要将高程图（Elevation Maps）或点云信息实时转化为运动规划，以应对楼梯、斜坡和障碍物。

## 主流技术路线

### 1. 基于模型的控制 (Model-based Control)
- **核心**：利用简化模型（如 单质点模型 CoM, 线性倒立摆 LIP）进行轨迹规划，配合全身控制（WBC）进行任务分解。
- **代表作**：MIT Cheetah 系列的变体，IHMC 的双足控制。

### 2. 层级强化学习 (Hierarchical RL)
- **核心**：分层架构，高层负责技能规划（Skill Planning），底层负责电机指令跟踪。
- **趋势**：通过奖励函数让机器人自主探索步态，解决非线性接触问题。

### 3. 生成式运动模型 (Generative Motion Models)
- **核心**：利用扩散模型（Diffusion Models）从人类数据中学习自然的运动先验。
- **进展**：ETH Zurich 的工作证明了扩散模型可以作为高效的实时全身运动生成器。

## 全身移动 (Whole-body Locomotion)

现代研究强调利用全身各个部位进行移动：
- **接触辅助**：在攀爬高箱时使用手臂辅助。
- **重心调节**：通过挥动手臂来补偿角动量。
- **环境自适应**：利用膝盖或身体侧面在狭窄空间支撑。

## 参考来源
- [sources/papers/eth-g1-diffusion.md](../../sources/papers/eth-g1-diffusion.md) — 基于扩散模型与 RL 的全身移动框架。
- [sources/papers/humanoid_hardware.md](../../sources/papers/humanoid_hardware.md) — 人形机器人硬件平台综述。

## 关联页面
- [Locomotion](./locomotion.md)
- [ZEST](../methods/zest.md) — Boston Dynamics 的跨形态高动态技能迁移框架
- [Diffusion-based Motion Generation](../methods/diffusion-motion-generation.md)
- [PPO](../methods/policy-optimization.md)
- [Whole-Body Coordination](../concepts/whole-body-coordination.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)
