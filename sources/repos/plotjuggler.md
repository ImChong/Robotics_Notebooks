# PlotJuggler

> 来源归档

- **标题：** PlotJuggler — The Time Series Visualization Tool that you deserve
- **类型：** repo / 桌面软件
- **来源：** Davide Faconti 等（PlotJuggler 组织）
- **链接：** https://github.com/PlotJuggler/PlotJuggler
- **官方站点：** https://plotjuggler.io/ — 归档见 [`sources/sites/plotjuggler-io.md`](../sites/plotjuggler-io.md)
- **Stars：** ~6.1k（2026-08）
- **入库日期：** 2026-06-17
- **最近复核：** 2026-08-10
- **许可证：** Mozilla Public License 2.0（闭源插件可开发）；部分第三方依赖含 LGPL（含 Qt）
- **一句话说明：** 跨平台 **时序 / 多模态** 可视化桌面工具：拖拽多曲线、离线文件与实时流、ROS1/ROS2 rosbag·topic、PX4 ULog、MCAP、MQTT/UDP/WebSocket 等；3.x 内置 Lua/Python 变换与插件架构；**PlotJuggler 4**（2026 起 beta）为从零重写，强调列式引擎、2D/3D 与 Extensions Marketplace。
- **沉淀到 wiki：** 是 → [`wiki/entities/plotjuggler.md`](../../wiki/entities/plotjuggler.md)

---

## 开源状态（步骤 2.5，截至 2026-08-10）

| 核查项 | 结论 |
|--------|------|
| 代码 | **已开源** — 主仓 + [`plotjuggler-ros-plugins`](https://github.com/PlotJuggler/plotjuggler-ros-plugins) 等插件仓 |
| 项目页 | [plotjuggler.io](https://plotjuggler.io/) 链到 GitHub / Releases |
| 发行线 | **3.17.x 稳定**（README 仍标 3.17）与 **PJ4 beta**（tag `3.999.x`，如 `3.999.3`）并存 |
| 可运行入口 | Releases（AppImage / 安装包 / Debian）、Snap、`ros-$ROS_DISTRO-plotjuggler-ros`、源码见 `COMPILE.md` |

---

## 安全提示（README 置顶）

官方 README **警告勿访问 plotjuggler.com**——该域名疑似钓鱼/恶意站点冒充项目；以 **[plotjuggler.io](https://plotjuggler.io/)**、GitHub 仓库与 Release 页为准。

---

## 版本线（复核摘要）

| 线 | 标识 | 要点 |
|----|------|------|
| **3.x 稳定** | README「PlotJuggler 3.17」；Release `3.17.2`（2026-05） | 拖拽时序、OpenGL、Lua/Python Function Editor、ROS/ULog/MCAP、Foxglove·PJ WebSocket bridge、ToolboxCSV/Parquet、Serial Port / Mosaico Flight 插件等 |
| **PlotJuggler 4 beta** | tag `3.999.0` / `3.999.3`；[公告](https://plotjuggler.io/blog/announcing-plotjuggler-4/)（2026-07-31） | **全新应用**（非 3.x 重构）：列式内存 + 大对象懒加载、多模态 2D/3D、Extensions Marketplace、新 UI、Mosaico 云源；beta2 强化 undo/redo、layout schema v4、点云/体素场景、Windows Qt IFW 安装包等 |

选型默认：生产 ROS/ULog 调试仍以 **3.17.x + ros 插件** 为主；评估多模态/3D/Marketplace 时跟 **PJ4 beta** Releases。

---

## 核心定位（README 3.17 + CHANGELOG）

- **交互**：Drag & Drop 选字段、多面板布局可保存复用；OpenGL 渲染，宣称可承载 **数千条时序、数百万点**。
- **离线**：CSV、[PX4 ULog](https://docs.px4.io/main/en/dev_log/ulog_file_format)、**MCAP**、自定义 DataLoader（如 CAN `.dbg`）；ULog 支持 **>2 GB** 文件（3.17）。
- **在线流**：MQTT、WebSockets、ZeroMQ、UDP、串口等；JSON / CBOR / BSON / MessagePack；与 **Foxglove / PlotJuggler websocket bridge** 互通任意 URL。
- **ROS**：打开 **rosbag** 与/或订阅 **ROS topic**（ROS1 与 ROS2）；OMG IDL / ROS 2 message definition；ROS 插件在独立仓 [`plotjuggler-ros-plugins`](https://github.com/PlotJuggler/plotjuggler-ros-plugins)。
- **LSL**：[Lab Streaming Layer](https://labstreaminglayer.readthedocs.io/info/intro.html) 设备流（独立插件仓）。
- **分析**：Transform Editor（导数、滑动平均、积分、RMS/MAV/stddev 等）；Custom Function Editor 用 **Lua** 与 **Python**（需 `python3-dev`）写多输入单输出脚本；ToolboxCSV/Parquet 导出。
- **构建**：`COMPILE.md` — Qt5 + CMake；可选 Apache Arrow（Parquet / Mosaico Flight）；AppImage 推荐 `appimage/build_in_docker.sh`。

## 安装与 ROS 集成（README 摘要）

| 渠道 | 说明 |
|------|------|
| GitHub Releases | Linux AppImage（x86/arm64）、macOS/Windows 安装包、Debian（bookworm/trixie）；PJ4 beta 另有 Windows Qt IFW 包 |
| Snap | `sudo snap install plotjuggler`（Ubuntu；含有限 ROS2 支持） |
| ROS apt | `sudo apt install ros-$ROS_DISTRO-plotjuggler-ros`；`ros2 run plotjuggler plotjuggler` |
| 源码 | 见仓库 `COMPILE.md` |

CI 覆盖 Windows、Ubuntu、macOS 及 ROS2 Humble / Jazzy / Rolling。

## 插件生态（节选）

| 插件 | 仓库 |
|------|------|
| ROS | [plotjuggler-ros-plugins](https://github.com/PlotJuggler/plotjuggler-ros-plugins) |
| MQTT | [plotjuggler-mqtt](https://github.com/PlotJuggler/plotjuggler-mqtt) |
| LSL | [plotjuggler-lsl](https://github.com/PlotJuggler/plotjuggler-lsl) |
| CAN .dbg | [plotjuggler-CAN-dbs](https://github.com/PlotJuggler/plotjuggler-CAN-dbs) |
| 示例 | [plotjuggler-sample-plugins](https://github.com/PlotJuggler/plotjuggler-sample-plugins) |

## 对 wiki 的映射

- [PlotJuggler（实体页）](../../wiki/entities/plotjuggler.md) — 机器人调试中的时序/多模态可视化选型
- [robot-policy-debug-playbook](../../wiki/queries/robot-policy-debug-playbook.md) — obs/action 与真机 log 对比
- [ROS 2 基础](../../wiki/concepts/ros2-basics.md) — topic / bag 调试工具链
- [PX4 Autopilot](../../wiki/entities/px4-autopilot.md) — ULog 飞行日志分析
- [Foxglove](../../wiki/entities/foxglove-studio.md) / [rerun](../../wiki/entities/rerun-io.md) / [MCAP](../../wiki/entities/mcap-log-format.md) — 多模态日志与可视化生态对照

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [plotjuggler-io.md](../sites/plotjuggler-io.md) | 官方站与 PJ4 公告 |
| [px4_autopilot.md](px4_autopilot.md) | ULog 为 PlotJuggler 一等公民格式 |
| [foxglove-studio.md](foxglove-studio.md) / [mcap-log-format.md](mcap-log-format.md) | WebSocket bridge / MCAP 互通 |
| [navigation2.md](navigation2.md) / [unitree_ros.md](unitree_ros.md) | ROS 栈真机/仿真 log 常用 PJ 打开 rosbag |
| [robot-policy-debug-playbook](../../wiki/queries/robot-policy-debug-playbook.md) | 与 rerun.io、Matplotlib 并列的时序对比工具 |
