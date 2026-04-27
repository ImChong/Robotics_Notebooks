# HumanoidSoccer (PAiD)

> 来源归档

- **标题：** HumanoidSoccer (Learning Soccer Skills for Humanoid Robots)
- **类型：** repo / paper-code
- **来源：** TeleHuman (Research Group)
- **链接：** https://github.com/TeleHuman/HumanoidSoccer
- **入库日期：** 2026-04-27
- **一句话说明：** 基于 PAiD 框架的人形机器人足球技能学习系统，支持 Unitree G1 机器人的类人化踢球动作。
- **沉淀到 wiki：** [paid-framework](../../wiki/methods/paid-framework.md), [humanoid-soccer](../../wiki/tasks/humanoid-soccer.md)

---

## 核心定位

该项目是论文 *"Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework"* 的官方实现。提出了 **PAiD (Perception-Action integrated Decision-making)** 框架。

---

## 关键技术栈

- **仿真引擎：** **Isaac Lab** (v2.1.1)
- **RL 框架：** **RSL_RL** (高性能机器人 RL 库)
- **网络架构：** 使用 **RNN** (处理感知与动作的时间序列依赖)
- **支持硬件：** **Unitree G1**

---

## PAiD 三阶段渐进式训练

1. **第一阶段：动作技能习得 (Skill Acquisition)**
   - 从人类动作数据中通过模仿学习（Imitation Learning）获取基础踢球技能。
2. **第二阶段：感知集成 (Perception Integration)**
   - 将视觉感知引入决策环路，增强动作在不同足球位置下的泛化能力。
3. **第三阶段：物理感知 Sim-to-Real (Physics-aware Transfer)**
   - 在仿真中加入复杂的物理扰动与延时模拟，确保在真机（Unitree G1）上的稳健表现。

---

## 技术特色

- **类人化踢球：** 不同于简单的碰撞，PAiD 追求更自然的挥腿与身体协调动作。
- **全向泛化：** 能够处理静止球与滚动球，支持室内外多种草坪环境。
- **视觉闭环：** 紧密集成视觉反馈，实现动态调整踢球力度与方向。

---

## 仓库结构

- `legged_gym/`: 修改版的环境定义，适配 Unitree G1 与足球任务。
- `resources/`: 包含人类踢球动作捕捉数据集。
- `rsl_rl/`: 核心强化学习算法代码。

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [unitree-g1.md](../../wiki/entities/unitree-g1.md) | 该框架的主要验证硬件平台 |
| [isaac-gym-isaac-lab.md](../../wiki/entities/isaac-gym-isaac-lab.md) | 使用最新的 Isaac Lab 仿真环境 |
| [imitation-learning.md](../../wiki/methods/imitation-learning.md) | PAiD 第一阶段使用了模仿学习技术 |
