# ROS 2：不同中间件 Vendor 与多 RMW 使用（一手文档）

> 来源归档

- **标题：** Different ROS 2 middleware vendors · Working with multiple RMW implementations
- **类型：** site（官方 Concepts / How-To；源在 `ros2/ros2_documentation`）
- **来源：** Open Robotics / ROS 2 社区
- **链接（Humble 文档 URL）：**
  - Vendors：https://docs.ros.org/en/humble/Concepts/Intermediate/About-Different-Middleware-Vendors.html
  - 多 RMW：https://docs.ros.org/en/humble/How-To-Guides/Working-with-multiple-RMW-implementations.html
- **源 RST（可离线对照；站点或有反爬）：**
  - https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Different-Middleware-Vendors.rst
  - https://github.com/ros2/ros2_documentation/blob/humble/source/How-To-Guides/Working-with-multiple-RMW-implementations.rst
- **入库日期：** 2026-07-28
- **一句话说明：** 官方定义 **RMW 包如何把 DDS/RTPS vendor 接到 ROS 2**、发行版支持矩阵，以及用 `RMW_IMPLEMENTATION` **运行时切换**与排错要点。
- **沉淀到 wiki：** 是 → [`wiki/concepts/rmw-interface.md`](../../wiki/concepts/rmw-interface.md)

## 为什么值得保留

- 与 [设计文](ros2-design-rmw-interface.md) 互补：设计讲「为何抽象」；本页讲「现在支持谁、怎么切」。
- 工程上最常踩的坑：daemon 仍用旧 RMW、跨 vendor 互通非保证、默认实现随 distro/`rmw_fastrtps_cpp` 可用性变化。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 文档源 | **已开源**（`ros2/ros2_documentation`，通常 Apache-2.0） |
| 渲染站 docs.ros.org | 公开；部分环境有 PoW 反爬，入库以 GitHub RST + 官方 URL 为准 |
| RMW / vendor 代码 | 见 [repos/rmw.md](../repos/rmw.md)、[fast-dds](../repos/fast-dds.md)、[cyclonedds](../repos/cyclonedds.md) |

## Vendors 页核心摘录（Humble 源）

- ROS 2 建立在 **DDS/RTPS** 之上：发现、序列化、传输；动机另见 Design《ROS on DDS》。
- 要用某一 DDS/RTPS，须有对应 **RMW** 包：用该实现的 API/工具实现抽象 ROS middleware 接口。
- 支持多实现是为了 **不绑死单一 vendor**（许可、平台、算力 footprint 等不同）。

### 支持矩阵（文档表；以当前 distro 页为准）

| 产品 | 许可（文档表述） | RMW 包 | 状态（文档表述） |
|------|------------------|--------|------------------|
| eProsima Fast DDS | Apache 2 | `rmw_fastrtps_cpp` | Full support；常为默认；随二进制发行 |
| Eclipse Cyclone DDS | EPL v2.0 | `rmw_cyclonedds_cpp` | Full support；随二进制发行 |
| RTI Connext DDS | commercial / research | `rmw_connextdds` | Full support；二进制含支持，Connext 需另装 |
| GurumNetworks GurumDDS | commercial | `rmw_gurumdds_cpp` | Community support；二进制含支持，GurumDDS 需另装 |

### 默认选择规则（文档）

- 多 RMW 共存且 Fast DDS 可用 → **默认 Fast DDS**。
- 无 Fast DDS → 按 RMW **实现标识符字母序** 选第一个（标识符即提供实现的 ROS 包名，如 `rmw_cyclonedds_cpp`）。
- 各 distro 默认亦见 [REP-2000](https://reps.openrobotics.org/rep-2000/)。

### 跨 vendor 互通

- **不保证**全场景互通；建议全系统统一 ROS 版本与同一 RMW。
- 文档列举的不支持配置示例：Fast DDS ↔ Connext（macOS 上 `WString`）、Connext ↔ Cyclone（`WString` pub/sub）等。

## 多 RMW How-To 核心摘录

- 环境变量 **`RMW_IMPLEMENTATION`**：设为 `rmw_cyclonedds_cpp`、`rmw_fastrtps_cpp`、`rmw_connextdds`、`rmw_gurumdds_cpp` 等。
- 源码工作区新增 DDS 后常需 **`--cmake-clean-cache`** 重建，使对应 RMW 包重新检测依赖。
- **排错：**
  - 未设置 `RMW_IMPLEMENTATION` → 用该 distro 默认。
  - 请求了未安装的实现 → 标识符不匹配 / not installed 类错误。
  - 切换 RMW 前 **`ros2 daemon stop`**：否则 CLI/`ros2 node` 可能仍连到旧 RMW 的 daemon。

## 对 wiki 的映射

- 主概念：[rmw-interface](../../wiki/concepts/rmw-interface.md)
- Vendor 实体：[fast-dds](../../wiki/entities/fast-dds.md)、[cyclone-dds](../../wiki/entities/cyclone-dds.md)
- 栈入口：[ros2-basics](../../wiki/concepts/ros2-basics.md)、[dds-communication](../../wiki/concepts/dds-communication.md)
- 设计动机：[ros2-design-rmw-interface.md](ros2-design-rmw-interface.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [ros2-official-documentation.md](ros2-official-documentation.md) | 文档站总归档；本页是 RMW 专题拆页 |
| [repos/rmw.md](../repos/rmw.md) | 抽象接口 C API |
| [dds_omg_rtos_edge_ota_safety_primary_refs.md](dds_omg_rtos_edge_ota_safety_primary_refs.md) | 合集索引链到本专题 |
