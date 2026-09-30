# Arena-Rosnav（GitHub 组织与主仓）

> 来源归档（ingest）

- **标题：** Arena-Rosnav — ROS 2 Social Navigation Development & Benchmarking
- **类型：** repo（组织 + 主入口仓）
- **组织：** <https://github.com/Arena-Rosnav>
- **主仓：** <https://github.com/Arena-Rosnav/arena-rosnav>
- **文档：** <https://docs.arena-rosnav.org/>
- **项目页（5.0）：** <https://5.arena-rosnav.org/>
- **许可：** MIT（`arena-rosnav` 仓 API 字段，2026-09-30）
- **入库日期：** 2026-09-30
- **一句话说明：** **ROS 2** 社交导航 **开发与评测平台**：自动安装脚本、多仿真后端、行人/机器人 spawn、**Move Base Flex** 集成、**stable-baselines3** RL 训练管线与 **arena-evaluation** 指标分析。

## 开源状态（GitHub 核查，2026-09-30）

| 项 | 状态 |
|----|------|
| 核心平台 | **已开源** — 组织下 40+ 仓，MIT 等许可以各仓为准 |
| 一键安装 | **已开源** — 文档 installation 教程 |
| 评测包 | **已开源** — [arena-evaluation](https://github.com/Arena-Rosnav/arena-evaluation) |
| DRL 规划器 | **已开源** — [rosnav-rl](https://github.com/Arena-Rosnav/rosnav-rl) 及 CADRL/CrowdNav/SARL 等 ROS 封装 |
| Isaac 集成 | **已开源** — [arena-isaac](https://github.com/Arena-Rosnav/arena-isaac)（运行依赖 NVIDIA Isaac 栈） |
| Web 远程跑仿真 | **已开源** — [arena-rosnav-webrunner](https://github.com/Arena-Rosnav/arena-rosnav-webrunner) |

## 组织内关键仓（策展索引）

| 仓库 | 角色 |
|------|------|
| [arena-rosnav](https://github.com/Arena-Rosnav/arena-rosnav) | 主入口与文档指向 |
| [arena-simulation-setup](https://github.com/Arena-Rosnav/arena-simulation-setup) | 仿真环境装配 |
| [arena-tools](https://github.com/Arena-Rosnav/arena-tools) / [arena-utils](https://github.com/Arena-Rosnav/arena-utils) | 工具与公共包 |
| [task-generator](https://github.com/Arena-Rosnav/task-generator) | 场景/任务生成 |
| [arena-evaluation](https://github.com/Arena-Rosnav/arena-evaluation) | 评测与指标 |
| [rosnav-rl](https://github.com/Arena-Rosnav/rosnav-rl) | 自研 DRL 规划器 |
| [move_base_flex](https://github.com/Arena-Rosnav/move_base_flex) | MBF 在 Arena 生态内的集成 |
| [pedsim_ros](https://github.com/Arena-Rosnav/pedsim_ros) / [crowdsim](https://github.com/Arena-Rosnav/crowdsim) | 行人与 crowd 仿真 |
| [real-world-bench](https://github.com/Arena-Rosnav/real-world-bench) | 真机评测相关 |
| [Arena-Training](https://github.com/Arena-Rosnav/Arena-Training) | 训练管线 |

## 典型复现路径（README / 文档口径）

```
自动安装（docs installation）→ 选择仿真后端（Gazebo / Isaac / Unity / Flatland）
  → 加载世界与 task mode → 挂载 planner（经典 / DRL / MBF）
  → arena-evaluation 导出社交导航指标
```

## 对 wiki 的映射

- [wiki/entities/arena-rosnav.md](../../wiki/entities/arena-rosnav.md)
- [sources/sites/5-arena-rosnav.md](../sites/5-arena-rosnav.md)
