# Gazebo Sim 官方仓库 README 与 C++ 主循环源码通读

> 来源归档（原文通读 · 支撑 `wiki/entities/gazebo-sim.md`）

- **标题：** gazebosim/gz-sim（README、Folder Structure 与 `Server` / `SimulationRunner` 源码）
- **作者 / 来源：** Open Robotics / Intrinsic（Maintainer: `arjoc`；Apache-2.0）
- **类型：** repo README & C++ source code
- **链接：** <https://github.com/gazebosim/gz-sim>
- **入库日期：** 2026-07-30（2026-11-28 一手仓库 README 与 C++ 入口源码通读更新）
- **一句话说明：** 核读 `gazebosim/gz-sim` 官方 `README.md` 与 `sim_main.cc` / `Server.cc` / `SimulationRunner.cc` 核心调度源码，确认其基于 ECS（Entity-Component-System）+ `gz-physics` / `gz-rendering` / `gz-sensors` / `gz-transport` 插件架构，以 SDFormat 为原生世界与模型描述（URDF 导入自动转 SDF），Apache-2.0 开源可运行。
- **开源状态：** 已开源，可运行（`gz sim`，Apache-2.0 许可）
- **沉淀到 wiki：** 是 → [`wiki/entities/gazebo-sim.md`](../../wiki/entities/gazebo-sim.md)
- **通读：** 已完成
- **核对范围：** 通读 `gazebosim/gz-sim` 官方 `README.md`（171 行全文：Features、Install、Usage、Folder Structure、Versioning、License）及 `src/cmd/sim_main.cc`、`src/Server.cc`、`src/SimulationRunner.cc` 核心调度路径（2026-11-28 抓取 `main` / `gz-sim9` 树核对）。

## 核心摘录（均出自官方 README 与 C++ 源码）

1. **模块化架构与核心依赖栈（README Features）：**
   - **动力学（`gz-physics`）：** 抽象物理引擎插件接口，可接 DART（默认）、Bullet 等刚体动力学后端；
   - **渲染与传感器（`gz-rendering` + `gz-sensors`）：** 支持 OGRE v2 光影/纹理与激光雷达、2D/3D 相机、接触、力矩、IMU、GPS 及其噪声模型；
   - **通信与界面（`gz-transport` + `gz-gui`）：** 基于 TCP/IP 异步消息与服务实现 C/S 分离和 ROS 2（经 `ros_gz`）桥接；
   - **描述与资产：** 原生使用 **SDFormat（`.sdf`）** 与 Gazebo Fuel 模型库（URDF 载入后底层自动转换为 SDF）。
2. **运行时入口与 ECS 步进循环（C++ 源码核对）：**
   - CLI `src/cmd/sim_main.cc` 解析 SDF 与运行参数后构造 `gz::sim::Server(serverConfig)`；
   - `Server` 构造函数在 `src/Server.cc` 中调用 `LoadSdfRootHelper` 解析 SDF root、建立 `SimulationRunner` 并调用 `SetupTransport()` 初始化通信，再按需下载 Fuel 资产；
   - `Server::Run` 调用各世界的 `SimulationRunner::Run`，每个物理步执行 `SimulationRunner::Step(UpdateInfo)`（`src/SimulationRunner.cc`）：同步世界控制状态、发布统计、激活挂起系统插件、并在 `UpdateSystems()` 内按阶段依次驱动所有已加载系统插件的 `PreUpdate` → `Update` → `PostUpdate`，最后清理创建/移除的实体与组件标记。
3. **工程部署细节：** Windows 下当前 `gz sim` 不支持单进程同时拉起 server 与 GUI，需分两个终端以 `gz sim -s` 与 `gz sim -g` 分别启动；源码混合编译/二进制安装混用时需正确配置 `GZ_CONFIG_PATH`。

## 对 wiki 的映射

- [`wiki/entities/gazebo-sim.md`](../../wiki/entities/gazebo-sim.md)
- [`wiki/entities/humanoid-motion-intelligence.md`](../../wiki/entities/humanoid-motion-intelligence.md)
