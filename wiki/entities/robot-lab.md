---
type: entity
tags: [repo, framework, reinforcement-learning, isaac-lab, unitree]
status: complete
updated: 2026-04-23
related:
  - ./isaac-gym-isaac-lab.md
  - ./legged-gym.md
  - ./unitree.md
sources:
  - ../../sources/repos/robot_lab.md
summary: "robot_lab 是一个基于 NVIDIA IsaacLab 的机器人 RL 扩展训练框架，支持 26+ 机器人型号，旨在通过模块化解耦简化机器人学习任务的开发。"
---

# robot_lab (IsaacLab 扩展框架)

**robot_lab** 是由开发者 `fan-ziqi` 维护的一个建立在 NVIDIA **IsaacLab** 之上的强化学习 (RL) 扩展库。它允许用户在隔离的仓库中开发机器人资产、环境和任务，而不必直接修改 IsaacLab 的核心代码，极大地提高了开发效率和代码的可维护性。

## 核心定位

在机器人学习的工具链中，robot_lab 扮演了“生态适配层”的角色：
- **IsaacLab 核心** 提供物理仿真、传感器模拟和基础环境接口。
- **robot_lab** 提供具体的机器人 URDF/USD 资产配置、特定任务（如速度跟踪、动作模仿）的定义以及训练脚本。

## 关键特性

1. **广泛的硬件支持**：原生支持 26+ 种机器人，包括四足（Anymal D, Go2, A1）、轮足（Go2W, Tita）以及主流人形机器人（Unitree G1/H1, GR1, Xbot, [Booster T1](./booster-robocup-demo.md)）。
2. **模块化任务开发**：采用 `manager_based` 任务流，支持 locomotion 和复杂的动作模仿。
3. **RL 框架集成**：深度集成 `RSL-RL` 作为主训练器，同时实验性支持 `CusRL` 和 `SKRL`。
4. **特殊任务支持**：
   - **AMP Dance**：利用 SKRL 驱动的动作风格迁移。
   - **BeyondMimic**：人形机器人的高性能动作模仿框架。
   - **[[mimickit]]**：由 Xue Bin Peng 开发的下一代动作控制研究套件，与 robot_lab 在 Isaac Lab 生态中高度互补。

## 仓库结构规范

robot_lab 遵循 IsaacLab 的扩展规范：
- `assets/`：存放机器人描述文件（URDF/USD）及对应的配置。
- `tasks/`：包含 `manager_based` 和 `direct` 两类 RL 任务实现。
- `scripts/`：提供统一的训练 (`train.py`)、评估 (`play.py`) 和导出入口。

## 关联页面
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)
- [legged_gym](./legged-gym.md)
- [Unitree 品牌主页](./unitree.md)
- [MimicKit (运动模仿套件)](./mimickit.md)
- [强化学习 (Reinforcement Learning)](../methods/reinforcement-learning.md)

## 参考来源
- [sources/repos/robot_lab.md](../../sources/repos/robot_lab.md)
- [fan-ziqi/robot_lab GitHub Repo](https://github.com/fan-ziqi/robot_lab)
