# DimOS（dimensionalOS/dimos）

> 来源归档

- **标题：** DimOS（Dimensional）
- **类型：** repo
- **机构：** Dimensional（dimensionalOS）
- **链接：** <https://github.com/dimensionalOS/dimos>
- **官网：** <https://dimensionalos.com/>
- **Stars：** ~3647（2026-07-09）
- **入库日期：** 2026-07-09
- **一句话说明：** **DimOS** 是 Dimensional 推出的 **agent-native 物理空间操作系统**：纯 Python 模块 + Blueprint 编排，默认无需 ROS，即可在 Unitree Go2/G1、机械臂、无人机等平台上做导航、感知、空间记忆与 MCP/自然语言 agent 控制。
- **沉淀到 wiki：** [wiki/entities/dimensionalos-dimos.md](../../wiki/entities/dimensionalos-dimos.md)

> **命名注意：** 本仓库与 ICCV 2023 论文 **DIMOS**（[zkf1997/DIMOS](https://github.com/zkf1997/DIMOS)，室内人–场景运动合成）**无关**；后者见 [sources/repos/dimos.md](dimos.md) 与 [wiki/entities/paper-dimos-human-scene-motion-synthesis.md](../../wiki/entities/paper-dimos-human-scene-motion-synthesis.md)。

---

## 核心定位

Dimensional 自称「**下一代通用机器人 SDK 标准**」与「**物理空间的 agentive operating system**」：一次 `pip install` 后，用 **Python Module + Blueprint** 把感知流、导航、空间记忆、控制环与 **LLM Agent（MCP skills）** 串成可复用应用；强调 **no ROS required**，但传输层可选 **LCM / SHM / DDS / ROS 2** 互操作。

状态：**Pre-Release Beta**（README 标注）。

## 仓库概况（GitHub API，2026-07-09）

| 字段 | 值 |
|------|-----|
| 创建时间 | 2024-10-19 |
| 主语言 | Python（~7.8M bytes），辅以 Rust/C++ native |
| 许可 | 见仓库 LICENSE |
| 包管理 | `uv` + `pyproject.toml` extras（`base`, `unitree`, `sim`, `manipulation` 等） |
| Python | 3.12 |
| 部署 | Nix flakes、NixOS、Docker、Ubuntu 22.04/24.04、macOS（部分） |

## 架构要点

| 概念 | 说明 |
|------|------|
| **Module** | 机器人子系统单元；用类型化 `In[]` / `Out[]` 流与 `@rpc` 生命周期方法通信；消息类型对齐 ROS `sensor_msgs` / `geometry_msgs` 等 |
| **Blueprint** | 模块接线说明书；`autoconnect(...)` 按 `(name, type)` 自动连流，可 `transports()` 覆盖为 LCM/SHM/DDS/ROS2 |
| **Runfile / CLI** | `dimos run <blueprint>`、`--simulation`、`--replay`；daemon 模式 + `dimos agent-send` / `dimos mcp call` |
| **Agent + MCP** | `McpServer` / `McpClient` 模块；skills 暴露为 MCP tools（如 `relative_move`）；支持本地 Ollama |
| **多语言** | Python 为主；C++/Lua/TypeScript 经 **LCM** 互操作示例 |

## 能力模块（README / docs 摘要）

| 能力 | 要点 |
|------|------|
| **Navigation & Mapping** | SLAM、动态避障、路径规划、自主探索；DimOS native 与 ROS 双路径 |
| **Perception** | 检测器、3D 投影、VLM、音频处理 |
| **Spatial Memory** | 时空 RAG、动态记忆、物体定位与持久性 |
| **Agentive Control** | 自然语言指令（如 "hey Robot, go find the kitchen"）+ MCP |
| **Manipulation** | xArm、AgileX Piper 键盘遥操作等（`dimos[manipulation]`） |
| **Simulation** | MuJoCo 四足/人形（`dimos --simulation run unitree-go2` / `unitree-g1-sim`） |
| **Replay** | 无硬件回放录制的 Go2/无人机 session（LFS 数据） |

## 硬件支持矩阵（README，2026-07）

| 形态 | 平台 | 成熟度 |
|------|------|--------|
| 四足 | Unitree Go2 Pro/Air | stable |
| 四足 | Unitree B1 | experimental |
| 人形 | Unitree G1 | beta |
| 机械臂 | xArm、AgileX Piper | beta |
| 无人机 | MAVLink、DJI Mavic | alpha |
| 传感 | openFT 力矩传感器（独立仓） | experimental |

真机 Go2 经 **WebRTC + ROBOT_IP**；文档提供 `dimos[base,unitree]` 快速安装。

## 代表性 Run 命令

| 命令 | 作用 |
|------|------|
| `dimos --replay run unitree-go2` | 四足导航回放（SLAM、costmap、A*） |
| `dimos --replay --replay-db go2_bigoffice run unitree-go2-memory` | 四足时空记忆回放 |
| `dimos --simulation run unitree-go2-agentic` | 仿真四足 + MCP agent |
| `dimos --simulation run unitree-g1-sim` | MuJoCo 人形仿真 |
| `dimos --replay run drone-agentic` | 无人机 + LLM agent 回放 |
| `dimos run demo-camera` | 无硬件 webcam 演示 |

## 对 wiki 的映射

- 沉淀实体页：[wiki/entities/dimensionalos-dimos.md](../../wiki/entities/dimensionalos-dimos.md)
- 交叉更新：[wiki/concepts/ros2-basics.md](../../wiki/concepts/ros2-basics.md)（ROS-optional 新一代集成栈）
- 交叉更新：[wiki/comparisons/ros2-vs-lcm.md](../../wiki/comparisons/ros2-vs-lcm.md)（DimOS 默认 LCM 传输）
- 交叉更新：[wiki/entities/unitree-g1.md](../../wiki/entities/unitree-g1.md)（G1 beta 平台支持）
- 交叉更新：[wiki/entities/lerobot.md](../../wiki/entities/lerobot.md)（训练侧 vs 部署/Agent OS 分工）
- 交叉更新：[wiki/overview/navigation-slam-autonomy-stack.md](../../wiki/overview/navigation-slam-autonomy-stack.md)（ROS-optional 导航/agent 栈）

## 参考链接

- 仓库：<https://github.com/dimensionalOS/dimos>
- 官网：<https://dimensionalos.com/>
- 安装脚本：`curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash`
