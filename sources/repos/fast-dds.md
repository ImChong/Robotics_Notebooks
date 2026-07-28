# eProsima/Fast-DDS

> 来源归档

- **标题：** eProsima Fast DDS
- **类型：** repo
- **来源：** eProsima
- **链接：** https://github.com/eProsima/Fast-DDS
- **文档：** https://fast-dds.docs.eprosima.com/（归档：[sites/fast-dds-docs.md](../sites/fast-dds-docs.md)）
- **Homepage：** https://eprosima.com
- **Stars：** ~2.9k（2026-07）
- **Forks：** ~941
- **默认分支：** `master`
- **最新发行：** v3.6.2（2026-07-02）
- **许可证：** Apache-2.0
- **入库日期：** 2026-07-28
- **一句话说明：** OMG DDS / RTPS 的完整 C++ 实现；ROS 2 默认 RMW 之一（经 `rmw_fastrtps`）；双层 DDS+RTPS API、多传输与 Discovery Server。
- **沉淀到 wiki：** 是 → [`wiki/entities/fast-dds.md`](../../wiki/entities/fast-dds.md)

## 开源状态（2026-07-28）

**已开源**：核心库、RTPS、依赖（Fast CDR 等）与文档源均为公开；商业 **Fast DDS Pro** 为另售扩展（见文档站对照表）。

质量声明：仓库 `QUALITY.md` 声称对齐 ROS 2 **Quality Level 1**（REP-2004）。

## README 定位（摘要）

- 实现 OMG DDS；线协议为 **RTPS**（不可靠传输上的 pub/sub，如 UDP）。
- 暴露 **DDS 层**与更底层的 **RTPS Writer/Reader API**。
- 主特性：Best Effort/Reliable、即插即用发现、可换传输、双 API 层。
- 采用方：ROS 2 LTS 默认中间件；FIWARE Robotics 目录条目。

## 与 ROS 2 的衔接

| 组件 | 仓 / 角色 |
|------|-----------|
| Vendor | 本仓 `eProsima/Fast-DDS`（`ros2.repos` 钉定） |
| RMW | [ros2/rmw_fastrtps](https://github.com/ros2/rmw_fastrtps)（~0.2k★，Apache-2.0） |
| 环境变量 | 常见：`RMW_IMPLEMENTATION=rmw_fastrtps_cpp` |

## 安装与演示入口（官方）

| 方式 | 入口 |
|------|------|
| 文档安装 | https://fast-dds.docs.eprosima.com/en/latest/installation/binaries/binaries_linux.html |
| 平台支持 | 仓库 `PLATFORM_SUPPORT.md` |
| Suite Docker | eProsima downloads 页提供 Fast DDS Suite 镜像 |
| Releases | https://github.com/eProsima/Fast-DDS/releases |

## Topics（GitHub）

`cpp` · `dds` · `distributed-systems` · `fastdds` · `middleware` · `omg` · `robotics` · `ros2` · `rtps`

## 对 wiki 的映射

- [Fast DDS 实体](../../wiki/entities/fast-dds.md)
- [DDS 通信](../../wiki/concepts/dds-communication.md)
- [ROS 2 基础](../../wiki/concepts/ros2-basics.md)
- 文档站：[fast-dds-docs.md](../sites/fast-dds-docs.md)
- 规范：[omg-dds-spec.md](../sites/omg-dds-spec.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [cyclonedds.md](cyclonedds.md) | 另一主流 ROS 2 DDS vendor；可互操作但 QoS/发现配置需各自钉死 |
| [ros2.md](ros2.md) | 元仓 `ros2.repos` 拉取本仓 |
| [rmw 对照](../sites/ros2-official-documentation.md) | 发行版默认 RMW 随 distro 变化 |
