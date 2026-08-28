# OmniSim（omnilink-tech/omnisim）

> 来源归档（repo · OmniLink 面向编码代理的开源机器人仿真器）

- **标题：** OmniSim — The simulator you can talk to
- **类型：** repo
- **来源：** OmniLink（[omnilink-tech](https://github.com/omnilink-tech)）
- **链接：** <https://github.com/omnilink-tech/omnisim>
- **项目页：** <https://www.omnilink-agents.com/omnisim> — 归档见 [`sources/sites/omnilink-agents-omnisim.md`](../sites/omnilink-agents-omnisim.md)
- **协议规范：** <https://github.com/omnilink-tech/omnisim/blob/main/PROTOCOL.md>（`omnisim_wire` 1.0）
- **代理入口：** <https://github.com/omnilink-tech/omnisim/blob/main/AGENTS.md>
- **许可证：** Apache-2.0（代码）；OmniSim 名称与 orb 商标受保护（`TRADEMARKS.md`）
- **Homepage 字段：** <https://www.omnilink-agents.com/omnisim>
- **默认分支：** `main`
- **入库日期：** 2026-08-28
- **一句话说明：** Webots 的独立 Apache-2.0 fork：面向编码代理，用 HTTP/JSON + 一等 MCP 驱动场景与机器人；物理后端现为 **Newton 唯一**（ODE 已删除），默认 MuJoCo Warp，wgpu 实时光栅渲染。
- **沉淀到 wiki：** 是 → [`wiki/entities/omnisim.md`](../../wiki/entities/omnisim.md)

## 开源状态（步骤 2.5，截至 2026-08-28）

项目页与 GitHub 均明确导向公开仓库；GitHub API：`license.spdx_id = Apache-2.0`，`stargazers_count = 31`，`homepage` 指向项目页，`pushed_at = 2026-08-26`。

| 资源 | 状态 |
|------|------|
| 仿真器源码 / 世界 / 控制器 / HTTP harness | **已开源**（Apache-2.0；Windows 有可下载 beta 包，Linux 为源码构建） |
| 一等 MCP server | **已开源**（`packages/omnisim-mcp/`，stdio，官方称 18 tools） |
| ROS 2 sidecar | **已开源但能力不完整**（`packages/omnisim-ros2/`：`simulation_interfaces` + 部分话题；`ros2_control` 仅核过差速底盘） |
| RL / Shadowing / BATON 技能库 | **仓库内可运行**（`python -m omnisim policy …`；人形演示多带承重吊索） |
| 训练容器 | **部分**：GHCR `omnisim-train` CUDA 训练镜像；尚无 demo 镜像 |
| 权重 / 真机迁移 | **无宣称的 sim-to-real**；Twin Shadow 协议面为保留、未实现 |
| macOS 物理 | **未验证**；无第二物理后端可回退 |

**结论：确认已开源（Apache-2.0）。** 可复现范围是「本机编译 + 官方 demo / 代理 harness」，不是「工业 ROS 全栈」或「已验证真机迁移」。

## 项目页与 README 的差异（须分开读）

营销页（2026-08-28 抓取）仍写：Newton GPU + **ODE CPU 回退**、`.wbt` 世界、以及含 Spot / Digit / Valkyrie / Franka 等的「18 机器人 / 11 厂商」名单。

仓库 README + `AGENTS.md`（与 `pushed_at` 对齐）则写：

- **Newton 是唯一物理后端**；ODE 于 2026-08-08（commit `bdc02139`）删除，`physicsBackend="ode"` 的 Solid **不再进求解器**。
- 世界扩展名以 **`.omniworld` 为准**（`.wbt` 为历史/双读规则）。
- README 机器人表为 OmniArm 6/7、OmniTug 500（仅视觉无碰撞体）、Unitree Go2/B2/G1/H1、OmniQuad、Husky/Jackal、UR3e–UR10e、Rosbot/XL、TurtleBot3、Mavic 2 Pro。

**以仓库 README / `AGENTS.md` 为事实源**；项目页作产品定位与 clone 入口，不单独采信其物理回退与机型名单。

## 仓库与能力要点（README / PROTOCOL / DEMOS 核对）

### 定位

- 口号：*The simulator you can talk to.* 编码代理读根目录 `AGENTS.md`，用自然语言装环境、建世界、接控制器。
- 独立 fork of [Webots](https://github.com/cyberbotics/webots)（Cyberbotics，2018 起 Apache-2.0）；不隶属、未经 Cyberbotics 背书。
- 自称「由代理在人类指导下编写」HTTP harness、Newton 集成、布料/软体、RL 管线与 ROS 2 sidecar。

### 代理面（PROTOCOL.md · `omnisim_wire` 1.0）

| Surface | 默认端口 | 作用 |
|---------|----------|------|
| Robot Bridge | `8765`（旧单臂 `6060`） | 每台可控机器人一条 HTTP/JSON 桥 |
| World Harness | `6789`（supervisor IPC `6790`） | 加载世界、热重载、场景树、截图、事件流 |
| Capture Service | `6791` | 高分辨率静帧 / 相机路径 / 编码 |
| Twin Shadow | （保留） | **未实现**；数字孪生硬对齐关节 |

传输：loopback HTTP/1.1 + JSON；角度弧度、位置米、世界系右手 ENU。README 对比表自称 **34 harness 端点 + 15 capture 动词**、**54 条加载诊断码**、**10 类运行时事件**。

### 物理与渲染

- **Newton 1.5.0** 为唯一后端；驱动 **MuJoCo（Warp）+ VBD** 两套求解器（README vs Isaac Sim 6.0.1 的对比表）。
- 默认 CPU 求解；GPU 路径需要 NVIDIA/CUDA（`mujoco_warp`）。无 CUDA 时轨迹自称与 GPU-visible 运行 **bit-identical**（两机测量）。
- 布料：Newton VBD；FEM 软体：tet-`SoftBody`；颗粒耦合「currently dead」；流体无。
- 渲染：wgpu（Vulkan / D3D12 / Metal），烘焙辐照探针，**非光追写实**。
- 约束缓冲：`mujoco_warp` 默认可静默丢弃超过 256 行的约束（16 辆驱动 rover 峰值 336/328）；需提高 `WorldInfo.newtonNjmax`。

### 吞吐（厂商自测，按机分开、不平均）

README 引用 2026-08-17 OmniBench 三机战役（1 env-step = 16 ms 控制步 = 8 物理子步）：

| GPU | GPU-batched physics @4096 | 引擎内 PPO @4096 |
|-----|---------------------------|------------------|
| RTX 3060 Laptop 6 GB | 165,369 env-steps/s | 98,136 |
| RTX 4000 Ada 20 GB | 280,820 | 201,850 |
| RTX 4090 24 GB | 535,377 | 499,734 |

相对 raw MuJoCo-Warp 开销约 **1.24–1.44×**。竞争单元格来自对方文档，**非** OmniSim 实测对方引擎。RAM/VRAM 足迹未公布。

### CLI 入口

```text
python -m omnisim doctor          # 本 clone 的真实二进制 / ABI / 端口 / 世界
python -m omnisim run-world <world>
python -m omnisim run-headless --until-finalized --fail-on-warning
python -m omnisim harness         # World Harness
python -m omnisim policy list | sequence <name>
```

首次构建：Windows `build_omni.bat`，Linux `bash scripts/install/linux_bootstrap.sh`；README 称 5–25 分钟。公共 beta：Windows 有包，Linux 为已验证源码构建，macOS 物理不在 beta 范围。

### 足式 / 人形 RL

- **Shadowing**：先验证参考轨迹可行，再在同一求解器上训练与部署（train == deploy）。
- **BATON**：把技能编成序列（如 G1 box delivery）。
- **关键披露**：官方 G1「象样走路」演示跑在承重平衡吊索上（`HARNESS_LAM0=0.9`、`HARNESS_KZ=2000`，骨盆向上力可达约 700 N）；楼梯 demo 将 `HARNESS_KZ=0`。Unitree 原策略在 OmniSim 中重托管可无吊索行走——引擎能承载，**自训出无辅助行走仍是开放问题**。

### 公开自承的短板（README「What OmniSim is worse at」）

- ROS 2 新且不完整：MoveIt 不可用（关节指令当 goal，插值中再来 setpoint 回 `409 busy`）；Nav2 从未对接；`Gyro`/`Accelerometer` 无可用数据；树中机器人无相机。
- Sim-to-real **零**真机迁移。
- 单 GPU 训练；无多卡/多机。
- 能力矩阵：45 probes 中约 78% 可用（两机）；恢复系数未实现；删除节点仍碰撞、新 spawn 节点不进求解器；复合 collider 默认只保留第一子体。
- 生态小于 Gazebo。

## 对 wiki 的映射

- 实体页 [`wiki/entities/omnisim.md`](../../wiki/entities/omnisim.md)
- 上游 fork [`wiki/entities/webots.md`](../../wiki/entities/webots.md) · [`sources/repos/webots.md`](webots.md)
- 物理引擎 [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)
- 选型 [`wiki/queries/simulator-selection-guide.md`](../../wiki/queries/simulator-selection-guide.md)
- MCP 协议 [`wiki/concepts/model-context-protocol.md`](../../wiki/concepts/model-context-protocol.md)
