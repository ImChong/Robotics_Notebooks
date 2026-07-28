# ros2/rmw

> 来源归档

- **标题：** ros2/rmw — The ROS Middleware (rmw) Interface
- **类型：** repo
- **来源：** Open Source Robotics Foundation / ROS 2 社区
- **链接：** https://github.com/ros2/rmw
- **组织：** https://github.com/ros2（归档：[sites/ros2-github-org.md](../sites/ros2-github-org.md)）
- **API 文档：** https://docs.ros.org/en/rolling/p/rmw/generated/（亦可：https://docs.ros2.org/latest/api/rmw/）
- **设计说明：** https://design.ros2.org/articles/ros_middleware_interface.html（归档：[sites/ros2-design-rmw-interface.md](../sites/ros2-design-rmw-interface.md)）
- **Stars：** ~119（2026-07）
- **默认分支：** `rolling`
- **许可证：** Apache-2.0
- **入库日期：** 2026-07-28
- **一句话说明：** 定义 ROS 2 **中间件抽象层 C API**（`rmw` 包）与实现者工具：节点/发布订阅/服务、wait set、图内省、错误与分配器等；具体 DDS vendor 实现在独立 `rmw_*` 仓。
- **沉淀到 wiki：** 是 → [`wiki/concepts/rmw-interface.md`](../../wiki/concepts/rmw-interface.md)

## 开源状态（2026-07-28）

**已开源**：接口头文件、CMake 辅助、质量声明与安全相关公共包均可公开 clone。本仓 **不包含** Fast DDS / Cyclone 等 vendor 本体，也不含各 `rmw_fastrtps_cpp` 等适配实现（见 sibling 仓）。

## README 定位（摘要）

> The ROS 2 Middleware Interface provides an abstraction layer to different DDS implementations for communication with the ROS 2 Client Library. This package contains the `rmw` interface for DDS implementation and some general functionality useful for implementers.

- 设计动机 → design 文  
- 接口细节 → API docs  
- **Quality Level 1**（见仓内 `QUALITY_DECLARATION.md`）

## 仓库内容结构（Rolling，2026-07）

| 路径 | 作用 |
|------|------|
| `rmw/` | **核心**：C 头与公共实现片段；主入口 `include/rmw/rmw.h`、`types.h` |
| `rmw_implementation_cmake/` | 供 RMW 实现包使用的 CMake 基础设施 |
| `rmw_security_common/` | 安全相关公共能力 |
| 根 `README.md` / `LICENSE` / `CONTRIBUTING.md` | 导航与许可 |

### `rmw` API 组件（头文件导读，来自 `rmw.h` 总览）

| 类别 | 内容 |
|------|------|
| 生命周期 | init / shutdown |
| 通信原语 | Node、Publisher、Subscription、Service Client/Server |
| 名称校验 | topic/service 全名、node name/namespace |
| 等待与唤醒 | wait sets、guard conditions |
| 图内省 | topic/service names & types、endpoint info |
| 基础设施 | allocators、error handling、macros、return codes、visibility |

## 相关 sibling（实现与加载，非本仓）

| 仓 / 包 | 角色 |
|---------|------|
| [ros2/rmw_implementation](https://github.com/ros2/rmw_implementation)（~25★，Apache-2.0） | 运行时/编译时选择具体 RMW；读 `RMW_IMPLEMENTATION` |
| [ros2/rmw_fastrtps](https://github.com/ros2/rmw_fastrtps) | Fast DDS 适配（`rmw_fastrtps_cpp`） |
| [ros2/rmw_cyclonedds](https://github.com/ros2/rmw_cyclonedds) | Cyclone DDS 适配 |
| [ros2/rmw_connextdds](https://github.com/ros2/rmw_connextdds) | RTI Connext 适配 |
| 元仓 `ros2.repos` | 钉定上述版本（归档：[ros2.md](ros2.md)） |

## 工程选用要点

1. 应用代码应停在 **rclcpp/rclpy**；只有写新 RMW 或深挖互通/性能时才直接读本仓头文件。
2. 部署侧用环境变量切换实现，见 [ros2-rmw-middleware-vendors.md](../sites/ros2-rmw-middleware-vendors.md)。
3. Unitree 等真机栈常钉 `rmw_cyclonedds_cpp`——与默认 Fast DDS 不同，须进仓库配置。

## 对 wiki 的映射

- [RMW 接口概念](../../wiki/concepts/rmw-interface.md)
- [ROS 2 基础](../../wiki/concepts/ros2-basics.md)
- [DDS 通信](../../wiki/concepts/dds-communication.md)
- [Fast DDS](../../wiki/entities/fast-dds.md) · [Cyclone DDS](../../wiki/entities/cyclone-dds.md)
- 设计文：[ros2-design-rmw-interface.md](../sites/ros2-design-rmw-interface.md)
- Vendor 文档：[ros2-rmw-middleware-vendors.md](../sites/ros2-rmw-middleware-vendors.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [ros2.md](ros2.md) | 元仓拉取本仓与各 `rmw_*` |
| [fast-dds.md](fast-dds.md) / [cyclonedds.md](cyclonedds.md) | RMW 之下的 vendor |
| [omg-dds-spec.md](../sites/omg-dds-spec.md) | DDS/RTPS 标准语义 |
