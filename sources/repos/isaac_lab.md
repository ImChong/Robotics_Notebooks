# isaac_lab

> 来源归档

- **标题：** NVIDIA Isaac Lab
- **类型：** repo
- **来源：** NVIDIA（isaac-sim 组织）
- **链接：** https://github.com/isaac-sim/IsaacLab
- **文档：** https://isaac-sim.github.io/IsaacLab/
- **入库日期：** 2026-07-21（从联合归档拆分为独立 source）
- **一句话说明：** 建立在 Isaac Sim 之上的官方 robot learning 框架；承接 IsaacGymEnvs / Orbit，提供 manager-based 与 direct 两套环境工作流。
- **代码：** https://github.com/isaac-sim/IsaacLab（已开源）
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-lab.md`](../../wiki/entities/isaac-lab.md)；教学任务 Cartpole 见 [`wiki/concepts/cartpole.md`](../../wiki/concepts/cartpole.md) 与 [`sources/sites/isaac-lab-cartpole.md`](../sites/isaac-lab-cartpole.md)；官方入门课见 [`sources/courses/nvidia_getting_started_isaac_lab.md`](../courses/nvidia_getting_started_isaac_lab.md)

---

## 核心定位

- **仿真底座：** 依赖 [Isaac Sim](../../wiki/entities/isaac-sim.md)（PhysX / USD / 传感器）
- **学习工作流：** `ManagerBasedRLEnv`（模块化 MDP）与 `DirectRLEnv`（单类实现）
- **训练后端：** RSL-RL、rl-games、SKRL 等

---

## Agent Skills（Isaac Lab 3.0）

官方仓库在 `skills/` 下维护 AI Agent 工作流说明。该目录包含用户工作流与开发者维护流程，不是运行时软件包；技能索引列出用途，Codex / Claude 通过仓库原生别名发现技能。

- v3.0.0-EA 发布说明记录 17 个用户技能，覆盖安装、迁移、环境构建、后端选择、RL 训练和调试、Newton 资产、传感器/执行器及 PhysX/Newton 策略迁移。
- [RL 调试技能](https://github.com/isaac-sim/IsaacLab/blob/main/skills/user/debug-rl-training/SKILL.md)要求先确认任务、环境契约和故障层，再做最小复现；同时对照 reward 与 task metrics，并优先一次只变更一个因素。
- [仿真间策略迁移技能](https://github.com/isaac-sim/IsaacLab/blob/main/skills/user/isaaclab-transferring-policies-sim-to-sim/SKILL.md)要求锁定策略输入输出契约、关节/刚体顺序、执行器行为，并验证 PhysX/Newton 的 PP/PN/NN/NP 组合。
- **文档**：<https://isaac-sim.github.io/IsaacLab/develop/source/developer-tools/agent_skills.html>
- **EA 发布说明**：<https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA>
- **Agent Skills 归档**：[`isaac-lab-agent-skills.md`](../sites/isaac-lab-agent-skills.md)

## 关联档案

- 默认任务清单：[`isaac_lab_environments.md`](./isaac_lab_environments.md)
- 官方入门课：[`nvidia_getting_started_isaac_lab.md`](../courses/nvidia_getting_started_isaac_lab.md)
- 联合索引：[`isaac_gym_isaac_lab.md`](./isaac_gym_isaac_lab.md)
- Isaac Sim：[`isaac_sim.md`](./isaac_sim.md)
- 前代 Gym：见联合归档与 [`wiki/entities/isaac-gym.md`](../../wiki/entities/isaac-gym.md)
