# ROS 2 Design：Middleware Interface（一手设计文档）

> 来源归档

- **标题：** ROS 2 middleware interface
- **类型：** site（ROS 2 Design 架构决策 / 设计说明）
- **来源：** Open Robotics / ROS 2 社区（作者 Dirk Thomas）
- **链接：** https://design.ros2.org/articles/ros_middleware_interface.html
- **源文：** https://github.com/ros2/design/blob/gh-pages/articles/060_ros_middleware_interface.md
- **撰写 / 修订：** 2014-08 撰写；2017-09 最后修改（页内标注）
- **入库日期：** 2026-07-28
- **一句话说明：** 解释为何在 ROS 客户端库与具体 DDS（或其它）实现之间引入 **抽象中间件接口（RMW）**：支持多 vendor、隐藏 DDS 细节、运行时切换实现。
- **沉淀到 wiki：** 是 → [`wiki/concepts/rmw-interface.md`](../../wiki/concepts/rmw-interface.md)

## 为什么值得保留

- 本库已有 [DDS 标准](omg-dds-spec.md) 与 [Fast/Cyclone 实现](../repos/fast-dds.md)，但缺少 **「ROS 为何不直连某一 DDS」** 的一手设计动机。
- 定义了分层：`user land → client library → middleware interface → adapter → DDS impl`。
- 明确 type support、opaque handle、运行时 vs 编译时选型等工程决策，是读 `ros2/rmw` 头文件前的概念入口。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 设计文档 | **公开可读**（design.ros2.org；源在 `ros2/design`） |
| 接口实现 | 见 [repos/rmw.md](../repos/rmw.md)（Apache-2.0） |

## 核心摘录

### 为何需要中间件接口

- ROS 1 通信基于自研协议（如 TCPROS）；ROS 2 选择建立在既有中间件（DDS）之上以复用成熟实现。
- 市面上 DDS 实现在平台、语言、性能、依赖与许可上差异大 → **不能绑死单一实现**。
- RMW 是 ROS 客户端库与具体实现之间的抽象；各实现通常是 **薄适配层**，把通用接口映射到 vendor API。

### 对 DDS 不可知（agnostic）

- 客户端库 **不向用户暴露 DDS 细节**，避免规范复杂度泄漏。
- 接口刻意少带 DDS 特有概念，以便将来用非 DDS 中间件（或拼装 discovery / serialization / pub-sub）实现同一接口。

### 信息流与 type support

- 接口之上只操作 **ROS 消息结构**；之下由实现把 ROS 对象转为中间件样本（或反向）。
- 类型映射由 **type support** 封装；可选静态 IDL 生成代码或 DynamicData / introspection（性能通常更差）。
- 也可绕过中间表示，直接对 ROS 消息做序列化/反序列化。

### 运行时切换

- 设计目标：多实现时尽量 **运行时选择**，避免为每个 vendor 维护整套二进制包（M×N）。
- 单实现构建时应满足「不用的能力不付费」：无额外多实现开销。
- 实际由 `rmw_implementation` 等机制按环境变量加载共享库（见官方 How-To）。

### 接口形态（设计期摘要）

| 原语 | 作用 |
|------|------|
| `create_node` / `create_publisher` / `publish` | 发布路径最小集合 |
| `get_type_support_handle` | 取与实现相关的类型信息（C 侧用宏名修饰） |
| opaque handles | 节点/发布者等句柄对用户不透明，带实现标识防串用 |
| 可选 native handles | 实现可另提供原生句柄以使用未暴露特性（会使用户代码绑定 vendor） |

### 概念映射（文档陈述）

| ROS | DDS（典型） |
|-----|-------------|
| 每个 ROS node | 一个 DDS DomainParticipant（同进程多节点 → 多 Participant） |
| publisher / subscriber | DDS publisher / subscriber；DataWriter/DataReader/Topic 不直接暴露给 ROS API |
| 部分 QoS | 映射到 DDS QoS；其余 DDS QoS 默认不经 ROS API 暴露 |

> 文档「Current implementation」小节中的 Connext/OpenSplice 包名偏历史；现行发行版 vendor 表以 [ros2-rmw-middleware-vendors.md](ros2-rmw-middleware-vendors.md) 为准。

## 对 wiki 的映射

- 主概念：[rmw-interface](../../wiki/concepts/rmw-interface.md)
- 上层栈：[ros2-basics](../../wiki/concepts/ros2-basics.md)
- 底层标准 / 实现：[dds-communication](../../wiki/concepts/dds-communication.md)、[fast-dds](../../wiki/entities/fast-dds.md)、[cyclone-dds](../../wiki/entities/cyclone-dds.md)
- 接口仓：[repos/rmw.md](../repos/rmw.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [ros2-rmw-middleware-vendors.md](ros2-rmw-middleware-vendors.md) | 发行版支持的 RMW 产品表与切换操作 |
| [ros2-official-documentation.md](ros2-official-documentation.md) | 文档站总入口；Design 站为其配套 |
| [repos/rmw.md](../repos/rmw.md) | C API 定义仓 |
| [omg-dds-spec.md](omg-dds-spec.md) | RMW 之下的 DDS/RTPS 标准层 |
