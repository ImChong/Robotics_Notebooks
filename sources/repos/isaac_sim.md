# isaac_sim

> 来源归档

- **标题：** NVIDIA Isaac Sim
- **类型：** repo / platform
- **来源：** NVIDIA（isaac-sim 组织）
- **链接：** https://github.com/isaac-sim/IsaacSim
- **文档：** https://docs.isaacsim.omniverse.nvidia.com/latest/index.html
- **许可证：** Apache 2.0（开源仿真栈，以仓库与文档为准）
- **入库日期：** 2026-07-21
- **一句话说明：** 基于 Omniverse / OpenUSD 的机器人仿真应用：导入资产、PhysX/Newton 物理、RTX 传感器、合成数据与 ROS 2 SIL；是 Isaac Lab 的仿真底座。
- **代码：** https://github.com/isaac-sim/IsaacSim（已开源）
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-sim.md`](../../wiki/entities/isaac-sim.md)

---

## Agent Skills（官方文档更新：2026-10-14）

Isaac Sim 官方文档将 Agent Skills 定义为按需加载的版本控制工作流指南，供 Cursor、Claude Code、Codex CLI 等 AI 编码 Agent 执行仿真任务。每项技能以 `SKILL.md` 为主体，可带辅助脚本与参考资料。

- 官方文档列出安装、运行中实例控制、机器人资产转换、物理配置、导航、操作、传感器、合成数据、渲染、ROS 2 等技能。
- 技能位于 Isaac Sim 仓库的 `skills/`，官方文档称它也随 pip 与二进制安装包分发；`skills/SKILLS.md` 是分类索引。
- 根目录的 `AGENTS.md`、`CLAUDE.md` 等指引文件帮助 Agent 发现技能。官方文档还提示，远程控制类技能可在 Kit 进程中执行任意 Python，应将服务绑定到 localhost 并保护运行环境。

- **Agent Skills 官方文档：** <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/development_tools/agent_skills.html>
- **技能目录与索引：** <https://github.com/isaac-sim/IsaacSim/tree/develop/skills>
- **索引文件：** <https://github.com/isaac-sim/IsaacSim/blob/develop/skills/SKILLS.md>
- **官方文档归档：** [`nvidia-isaac-sim-agent-skills.md`](../sites/nvidia-isaac-sim-agent-skills.md)

## 核心定位

Isaac Sim 解决的是「高保真机器人仿真工作台」问题，而不是 RL 训练框架本身：

- **场景与资产：** URDF / MJCF / CAD / USD 导入，共享 OpenUSD stage
- **物理与传感：** PhysX（及 Newton 路径）、RTX / 物理传感器
- **下游连接：** 为 Isaac Lab 准备机器人与场景；Replicator 合成数据；ROS 2 software-in-the-loop

与两代学习框架的分工：

| 产品 | 角色 |
|------|------|
| [Isaac Gym](../../wiki/entities/isaac-gym.md) | 早期独立 GPU 并行 RL 仿真（legacy） |
| **Isaac Sim** | Omniverse 机器人仿真应用 / 底座 |
| [Isaac Lab](../../wiki/entities/isaac-lab.md) | 跑在 Isaac Sim 上的 robot learning 框架 |

---

## 关键 Python 入口（对齐官方 API）

| 模块 / 类 | 作用 |
|-----------|------|
| `isaacsim.SimulationApp` | 启动 Omniverse Kit；须在导入其它 Isaac 模块前实例化 |
| `isaacsim.core.api.SimulationContext` | 仿真步进、物理场景与 timeline |
| `isaacsim.core.api.World` | 继承 SimulationContext，附加 Scene / 任务层 |
| Articulation / RigidPrim 等 | USD 机器人与刚体包装 |
| Replicator / ROS 2 扩展 | 合成数据与外部栈连接 |

---

## 关联档案

- 联合索引（历史归档）：[`isaac_gym_isaac_lab.md`](./isaac_gym_isaac_lab.md)
- Omniverse 底座概念：[wiki/entities/nvidia-omniverse.md](../../wiki/entities/nvidia-omniverse.md)
- 选型对比：[wiki/comparisons/mujoco-vs-isaac-sim.md](../../wiki/comparisons/mujoco-vs-isaac-sim.md)
