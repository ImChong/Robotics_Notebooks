# Robot Native Engine（rsasaki0109/RobotNativeEngine）

> 来源归档（开源仓库）

- **标题：** Robot Native Engine（RNE）
- **类型：** repo / Rust 机器人仿真与具身 AI 引擎
- **代码：** <https://github.com/rsasaki0109/RobotNativeEngine>
- **项目说明：** <https://github.com/rsasaki0109/RobotNativeEngine#readme>（仓库 README；未找到独立托管项目站）
- **架构文档：** <https://github.com/rsasaki0109/RobotNativeEngine/blob/main/docs/architecture/000_overview.md>
- **语言：** Rust 核心；提供 Python 接口、可选 ROS 2 adapter 和 WebAssembly 浏览器 Viewer
- **物理与渲染：** backend-neutral physics traits；Rapier 与 MuJoCo 后端；wgpu renderer；headless 仿真不要求启动 renderer
- **许可：** Apache-2.0 或 MIT，二选一
- **开源状态（README / 文档核查，2026-10-05）：** 核心源码、示例、文档和仓内资产公开。未确认独立发布的训练权重或数据集；仓库示例不等于通用策略 checkpoint。
- **入库日期：** 2026-10-05
- **一句话说明：** 用 Rust 构建的机器人原生仿真核心，将机器人、执行器、传感器、Agent 与 episode 作为一等实体，并强调固定步进、确定性回放、传感器模拟和可选渲染 / ROS 2 适配。

## 核查入口

| 入口 | 内容 |
|------|------|
| [README](https://github.com/rsasaki0109/RobotNativeEngine/blob/main/README.md) | 项目定位、功能总览、演示、架构、Quickstart 和开源许可 |
| [Architecture Overview](https://github.com/rsasaki0109/RobotNativeEngine/blob/main/docs/architecture/000_overview.md) | crate 分层与控制 / 物理 / 传感器 / 记录 / 渲染流程 |
| [Examples](https://github.com/rsasaki0109/RobotNativeEngine/blob/main/examples/README.md) | Hello world、URDF、Python policy、传感器、操控、Go2、G1 等示例目录 |
| [Robot Workbench](https://github.com/rsasaki0109/RobotNativeEngine/blob/main/docs/ROBOT_WORKBENCH.md) | 本地工作台：URDF / MJCF、关节控制、障碍编辑与 RGB-D / LiDAR 视图 |
| [G1 locomotion](https://github.com/rsasaki0109/RobotNativeEngine/blob/main/docs/G1_LOCOMOTION.md) | G1 平衡、仿真翻跟头与行走边界 |
| [Go2 door](https://github.com/rsasaki0109/RobotNativeEngine/blob/main/docs/GO2_DOOR.md) | Go2 开门任务、Mid-360 / SLAM 输入及场景特定评测 |
| [Livox Mid-360](https://github.com/rsasaki0109/RobotNativeEngine/blob/main/docs/LIVOX_MID360.md) | 传感器模型与 Go2 实测记录拟合 |
| [Web Viewer README](https://github.com/rsasaki0109/RobotNativeEngine/blob/main/web/rne_web_viewer/README.md) | WASM Viewer 本地构建与 replay 检查；不是已确认的在线 demo |

## 来源摘录与复现路径

1. README 将 RNE 定位为 deterministic simulation、embodied AI、synthetic sensors 和 policy evaluation 的 Rust 引擎；机器人、传感器、执行器、Agent 与 episode 属于仿真世界实体，ROS 2 位于可选 adapter，而非核心依赖。
2. 核心工作区包含 rne_core、rne_ecs、rne_world、rne_robot、rne_sensor、rne_ai、rne_data、rne_physics、rne_planning、rne_dynamics、rne_legged、rne_wbc 等 crate；rne_physics 提供后端抽象，具体实现与核心 API 解耦。
3. 固定 SimClock、显式 seed、稳定实体顺序和 replay digest 用于可复核的 headless 仿真。wgpu 渲染是可选支路，不能作为仿真逻辑推进的时钟。
4. Quickstart 从 hello_world 与 falling_cube 示例开始。robot_workbench 可加载 URDF / MJCF，在本地浏览器窗口操控关节并查看传感器画面。
5. 仓库展示包括 G1、Go2、OpenArm、移动操作和 PLATEAU 无人机等场景；Go2 开门文档报告特定场景的定位误差为 0.058 m RMS。该数字是该仿真任务的定位评测，不是 Mid-360 的通用精度规格。
6. G1 翻跟头为仿真中由参数搜索得到的控制结果，不是强化学习策略的展示；文档将该结果限定为 simulator result。当前资料不支持将其推广为 G1 真机能力。
7. Web Viewer 的 README 使用 trunk serve 本地运行，加载 .rne-replay / JSON artifact 展示时间轴和状态摘要；这不是仓库已托管的在线查看网站。

## 复现入口

从仓库根目录运行：

    git clone https://github.com/rsasaki0109/RobotNativeEngine.git
    cd RobotNativeEngine
    cargo run -p hello_world --example 00_hello_world
    cargo run -p falling_cube --example 01_falling_cube

机器人交互工作台：

    cargo run --release --locked -p robot_workbench

完整验证命令与各 smoke gate 见 README / xtask 文档；本次仅核查仓库文档，没有在目标环境编译或运行 Rust 示例。

## 对 wiki 的映射

- [Robot Native Engine 实体页](../../wiki/entities/robot-native-engine.md)
- [MuJoCo 实体页](../../wiki/entities/mujoco.md) — RNE 的可选物理后端之一
- [Sim2Real 概念页](../../wiki/concepts/sim2real.md) — 区分仿真回归证据与硬件迁移证据
