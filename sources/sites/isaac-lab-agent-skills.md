# Agent Skills — NVIDIA Isaac Lab

> 官方文档、技能源码与 3.0.0 Early Access 发布记录归档。

- **标题：** Agent Skills — Isaac Lab Documentation
- **类型：** 官方开发者文档 / 仓库内 AI Agent 工作流指南
- **文档：** <https://isaac-sim.github.io/IsaacLab/develop/source/developer-tools/agent_skills.html>
- **GitHub 仓库：** <https://github.com/isaac-sim/IsaacLab>
- **技能目录：** <https://github.com/isaac-sim/IsaacLab/tree/main/skills>
- **3.0 Early Access 发布说明：** <https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA>（2026-09-16 发布）
- **一句话说明：** Isaac Lab 在仓库中维护按任务加载的 Markdown 工作流，让编码 Agent 遵循安装、训练、调试、迁移和贡献流程。
- **技能属性：** 官方称其为仓库维护的指导材料，不是运行时 Python 包；用户技能与开发者技能分目录管理。
- **EA 技能数量：** v3.0.0-EA 发布说明记录 17 个用户技能。
- **迁移策略技能：** <https://github.com/isaac-sim/IsaacLab/blob/main/skills/user/isaaclab-transferring-policies-sim-to-sim/SKILL.md>
- **RL 调试技能：** <https://github.com/isaac-sim/IsaacLab/blob/main/skills/user/debug-rl-training/SKILL.md>
- **关联的 Isaac Sim Agent Skills 文档：** <https://docs.isaacsim.omniverse.nvidia.com/latest/development_tools/agent_skills.html>
- **沉淀到 wiki：** [`wiki/entities/isaac-lab.md`](../../wiki/entities/isaac-lab.md)

---

## 收录要点

- `SKILL.md` 是技能入口；用户技能可含 `evaluations.md`、参考资料与示例，开发者技能面向贡献和维护流程。
- RL 调试技能强调先核验环境契约和失败层，做小复现，联合查看奖励与任务指标，并一次只改变一个变量。
- 策略迁移技能要求控制同一任务的策略侧契约，核验关节/刚体排序与执行器行为，并评估 PhysX→PhysX、PhysX→Newton、Newton→Newton、Newton→PhysX 四种组合。
- EA 文档列出的迁移验证任务包括 Franka Lift、G1 rough locomotion、ANYmal-D rough locomotion；这不代表任意策略或任务都已验证。
- 用户给出的 GitHub `develop/skills` 路径在此次核查时无法解析（仓库 API 未找到 `develop` ref）；因此技能源码链接指向仓库当前 `main`，EA 发布内容可用 `v3.0.0-EA` tag 固定复核。

## 官方一手资料

1. [Agent Skills 官方说明（develop 文档）](https://isaac-sim.github.io/IsaacLab/develop/source/developer-tools/agent_skills.html)
2. [Isaac Lab 官方技能索引](https://github.com/isaac-sim/IsaacLab/blob/main/skills/README.md)
3. [Isaac Lab 3.0 Early Access 发布说明](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
4. [PhysX/Newton 仿真间策略迁移技能](https://github.com/isaac-sim/IsaacLab/blob/main/skills/user/isaaclab-transferring-policies-sim-to-sim/SKILL.md)
5. [Isaac Lab RL 训练调试技能](https://github.com/isaac-sim/IsaacLab/blob/main/skills/user/debug-rl-training/SKILL.md)
6. [Isaac Sim Agent Skills 官方文档](https://docs.isaacsim.omniverse.nvidia.com/latest/development_tools/agent_skills.html)

## 对本仓库的映射

- Isaac Lab 实体页：[wiki/entities/isaac-lab.md](../../wiki/entities/isaac-lab.md)
- Isaac Lab 仓库归档：[sources/repos/isaac_lab.md](../repos/isaac_lab.md)
- Isaac Sim 实体页：[wiki/entities/isaac-sim.md](../../wiki/entities/isaac-sim.md)
