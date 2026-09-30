# Arena 5.0 项目页（Arena-Rosnav）

> 来源归档（ingest）

- **标题：** Arena 5.0 — Photorealistic ROS2 Simulation for Social Navigation
- **类型：** site（项目页 + 演示 + Benchmark 叙事）
- **URL：** <https://5.arena-rosnav.org/>
- **论文（Arena 5.0）：** <https://5.arena-rosnav.org/arena5.pdf>（RSS 2025；项目页「Paper」链）
- **会议：** [RSS 2025](https://roboticsconference.org/program/papers/92/)
- **演示视频：** <https://www.youtube.com/watch?v=MrKrnHhk8IA>
- **文档：** <https://docs.arena-rosnav.org/>
- **机构：** 新加坡国立大学（NUS）、柏林工业大学（TUB）、慕尼黑工业大学（TUM）
- **入库日期：** 2026-09-30
- **一句话说明：** 第五代 **Arena-Rosnav** 开源栈：在 **ROS 2** 上统一 **场景生成、行人/机器人 spawn、多后端仿真（含 Isaac Sim 等）与社交导航评测指标**，面向人机共存环境下的算法开发与 SOTA 对比。

## 开源核查（步骤 2.5，2026-09-30）

| 资源 | 状态 | 说明 |
|------|------|------|
| 主仓与生态 | **已开源** | GitHub 组织 [Arena-Rosnav](https://github.com/Arena-Rosnav)；入口仓 [arena-rosnav](https://github.com/Arena-Rosnav/arena-rosnav)（MIT）；含 `arena-evaluation`、`rosnav-rl`、`move_base_flex` fork、`arena-isaac` 等 |
| 安装 | **已开源** | 文档 [automatic installation](https://docs.arena-rosnav.org/en/latest/tutorials/installation/) |
| Arena 5.0 论文 PDF | **已公开** | 托管于项目页 `arena5.pdf`（非 arXiv 链；前代 Arena 4.0 见 [arXiv:2409.12471](https://arxiv.org/abs/2409.12471)） |
| Isaac Sim / Gym | **依赖 NVIDIA 栈** | 照片级训练与仿真需 Isaac 系环境；Gazebo / Unity / Flatland 等后端见组织 README 锚点 |
| 预训练权重 | **随各 planner 子仓** | 如 `rosnav-rl`、CADRL/CrowdNav 等 ROS 封装；以各仓 README 为准 |

## 项目页主张（摘要）

1. **Isaac Gym / 照片级集成（5.0）：** 将 NVIDIA Isaac 系仿真接入 Arena 平台，保留随机环境生成、评测、ROS 2、规划器与机器人 API 等既有模块。
2. **SOTA 社交导航 Benchmark：** 多难度生成/定制世界与场景，用 **碰撞次数、到达时间、路径长度、个人空间停留、注视行人、被行人看见** 等指标评估效率与社交意识。
3. **场景与任务生成：** 扩展 emergency/rescue 等可定制社交导航场景与任务规划模块。

## 平台四块（Pipeline）

| 模块 | 作用 |
|------|------|
| Generate Worlds | 办公室、医院、食堂、仓库等预置场景 + 动态迷宫 |
| Spawn Humans | 社会力等行人模型 |
| Spawn Robots | 多机器人模型（含 Go1 四足等） |
| Benchmarking & Task Modes | `scenario` / `random` / `parametrized` / `guided` / `explore` 等任务模式 |

## 仿真后端（项目页列）

Isaac Sim、Unity、Gazebo、Flatland（各后端说明链至 GitHub 组织 README 锚点）。

## 对 wiki 的映射

- [wiki/entities/arena-rosnav.md](../../wiki/entities/arena-rosnav.md)
- [sources/repos/arena-rosnav.md](../repos/arena-rosnav.md)
- [sources/papers/arena_rosnav_lineage.md](../papers/arena_rosnav_lineage.md)
