# eclipse-cyclonedds/cyclonedds

> 来源归档

- **标题：** Eclipse Cyclone DDS
- **类型：** repo
- **来源：** Eclipse Foundation（Eclipse IoT `iot.cyclonedds`）
- **链接：** https://github.com/eclipse-cyclonedds/cyclonedds
- **文档：** https://cyclonedds.io/docs/（官网：[sites/cyclonedds-io.md](../sites/cyclonedds-io.md)）
- **Homepage：** https://projects.eclipse.org/projects/iot.cyclonedds
- **Stars：** ~1.3k（2026-07）
- **Forks：** ~471
- **默认分支：** `master`
- **最新发行：** 11.0.1（2026-03-20；发行称号 *Marche des "Davidsbündler" contre les Philistins*）
- **许可证：** EPL-2.0 **或** EDL-1.0（双许可；README badge 亦标注）
- **入库日期：** 2026-07-28
- **一句话说明：** 高性能开源 OMG DDS 实现（C 核心 + 独立 C++/Python 绑定仓）；ROS 2 **tier-1** 中间件；Unitree 等厂商真机栈常用。
- **沉淀到 wiki：** 是 → [`wiki/entities/cyclone-dds.md`](../../wiki/entities/cyclone-dds.md)

## 开源状态（2026-07-28）

**已开源**：核心 C API、构建系统、文档与发行包公开；语言绑定在 sibling：
- https://github.com/eclipse-cyclonedds/cyclonedds-cxx
- https://github.com/eclipse-cyclonedds/cyclonedds-python

## README 定位（摘要）

- 完整开源实现 OMG DDS；强调 **eventually consistent shared data space** 与无中心 broker。
- 规范覆盖（README 自述）：DCPS（Minimum / Ownership / 部分 Content）、DDS Security、C++ API、XTypes（有 caveats）、**DDSI-RTPS 2.5**。
- ROS 2：**tier-1 middleware**。
- 可选依赖：OpenSSL（安全）、**Eclipse Iceoryx** 2.0（共享内存 / 零拷贝）、CMake ≥ 3.16。

## 与 ROS 2 / 厂商的衔接

| 组件 | 仓 / 角色 |
|------|-----------|
| Vendor | 本仓（`ros2.repos` 钉定） |
| RMW | [ros2/rmw_cyclonedds](https://github.com/ros2/rmw_cyclonedds)（~0.17k★，Apache-2.0） |
| 环境变量 | 常见：`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| Unitree | [`unitree_ros2`](unitree_ros2.md) / SDK2 默认 CycloneDDS；Foxy 常需钉 **0.10.x** |

## 构建要点（README）

| 项 | 要求 |
|----|------|
| OS | Linux / macOS / Windows 10（另有 *BSD、QNX 等 caveats） |
| 工具 | C 编译器、CMake ≥ 3.16；可选 Git、OpenSSL、Iceoryx、Bison |
| IDL | C/C++ 通常经内置 IDL 编译器；Python 可动态定义类型 |

## 对 wiki 的映射

- [Cyclone DDS 实体](../../wiki/entities/cyclone-dds.md)
- [DDS 通信](../../wiki/concepts/dds-communication.md)
- [unitree_ros2](../../wiki/entities/unitree-ros2.md)
- [ROS 2 基础](../../wiki/concepts/ros2-basics.md)
- 官网：[cyclonedds-io.md](../sites/cyclonedds-io.md)
- 规范：[omg-dds-spec.md](../sites/omg-dds-spec.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [fast-dds.md](fast-dds.md) | 另一主流 vendor；同 RTPS 可互通 |
| [ros2.md](ros2.md) | 元仓拉取本仓 |
| [unitree_sdk2.md](unitree_sdk2.md) / [unitree_ros2.md](unitree_ros2.md) | 真机默认消费本实现 |
| [unitree_dds_wrapper.md](unitree_dds_wrapper.md) | 厂商侧 DDS 包装（若仍引用） |
