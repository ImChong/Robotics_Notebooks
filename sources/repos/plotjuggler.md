# PlotJuggler

> 来源归档

- **标题：** PlotJuggler
- **类型：** repo / 桌面软件
- **来源：** Davide Faconti 等（PlotJuggler 组织）
- **链接：** https://github.com/PlotJuggler/PlotJuggler
- **Stars：** ~6k（2026-06）
- **入库日期：** 2026-06-17
- **许可证：** Mozilla Public License 2.0（闭源插件可开发）；部分第三方依赖含 LGPL（含 Qt）
- **一句话说明：** 跨平台 **时序数据可视化** 桌面工具：拖拽式多曲线绘图、离线文件与实时流、ROS1/ROS2 rosbag 与 topic、PX4 ULog、MQTT/UDP/LSL 等；内置导数/积分/滑动平均与 Lua 自定义变换，插件可扩展。
- **沉淀到 wiki：** 是 → [`wiki/entities/plotjuggler.md`](../../wiki/entities/plotjuggler.md)

---

## 安全提示（README 置顶）

官方 README **警告勿访问 plotjuggler.com**——该域名疑似钓鱼/恶意站点冒充项目；以 GitHub 仓库与 Release 页为准。

---

## 核心定位（README 3.17）

- **交互**：Drag & Drop 选字段、多面板布局可保存复用；OpenGL 渲染，宣称可承载 **数千条时序、数百万点**。
- **离线**：CSV、[PX4 ULog](https://docs.px4.io/main/en/dev_log/ulog_file_format)、自定义 DataLoader 插件（如 CAN `.dbg`）。
- **在线流**：MQTT、WebSockets、ZeroMQ、UDP 等；JSON / CBOR / BSON / MessagePack 等格式。
- **ROS**：打开 **rosbag** 与/或订阅 **ROS topic**（ROS1 与 ROS2）；ROS 插件在独立仓 [`plotjuggler-ros-plugins`](https://github.com/PlotJuggler/plotjuggler-ros-plugins)。
- **LSL**：[Lab Streaming Layer](https://labstreaminglayer.readthedocs.io/info/intro.html) 设备流（独立插件仓）。
- **分析**：Transform Editor（导数、滑动平均、积分等）；Custom Function Editor 用 **Lua** 写多输入单输出脚本。

## 安装与 ROS 集成（README 摘要）

| 渠道 | 说明 |
|------|------|
| GitHub Releases | Linux AppImage、macOS/Windows 安装包、Debian 包 |
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

- [PlotJuggler（实体页）](../../wiki/entities/plotjuggler.md) — 机器人调试中的时序可视化选型
- [robot-policy-debug-playbook](../../wiki/queries/robot-policy-debug-playbook.md) — obs/action 与真机 log 对比
- [ROS 2 基础](../../wiki/concepts/ros2-basics.md) — topic / bag 调试工具链
- [PX4 Autopilot](../../wiki/entities/px4-autopilot.md) — ULog 飞行日志分析

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [px4_autopilot.md](px4_autopilot.md) | ULog 为 PlotJuggler 一等公民格式 |
| [navigation2.md](navigation2.md) / [unitree_ros.md](unitree_ros.md) | ROS 栈真机/仿真 log 常用 PJ 打开 rosbag |
| [robot-policy-debug-playbook](../../wiki/queries/robot-policy-debug-playbook.md) | 与 rerun.io、Matplotlib 并列的时序对比工具 |
