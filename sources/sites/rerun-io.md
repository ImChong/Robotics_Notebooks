# Rerun 官方站点与文档

> 官方站点与产品文档归档

- **标题：** Rerun — The Data Layer for Physical AI
- **类型：** 官方站点 / 文档 / 技术博客
- **开发方：** Rerun Technologies
- **链接：** https://rerun.io/
- **文档：** https://rerun.io/docs/
- **入库日期：** 2026-10-04
- **开源状态：** 文档对应的 Rerun SDK 与 Viewer 代码位于公开 GitHub 仓库；Rerun Hub 为商业化数据目录与后端服务。
- **对应仓库：** [rerun-io/rerun](../repos/rerun-io.md)
- **一句话说明：** Rerun 官方产品说明、开发文档、机器人示例和技术博客，覆盖从多模态日志到可视化、查询、预处理与训练数据准备的工作流。
- **沉淀到 wiki：** 是 → [Rerun 实体页](../../wiki/entities/rerun-io.md)

## 官方资料索引

| 主题 | 一手资料 | 内容 |
|------|----------|------|
| 产品定位 | https://rerun.io/docs/overview/what-is-rerun | SDK、Viewer 与 Hub 的组成，以及统一数据层的定位 |
| 高层架构 | https://rerun.io/docs/concepts/how-does-rerun-work | Logging SDK、Viewer、存储与数据流 |
| 时间轴回放 | https://rerun.io/docs/reference/viewer/timeline | 多时间轴选择、播放控制、逐步查看与时间拖动 |
| MCAP 与 ROS 2 | https://rerun.io/docs/howto/logging-and-ingestion/mcap | MCAP 导入、ROS 2 消息解码、RRD 转换 |
| URDF | https://rerun.io/docs/howto/logging-and-ingestion/urdf | 机器人模型、网格资源与关节 frame 导入 |
| ROS 2 示例 | https://rerun.io/examples/robotics/ros_node | 订阅 ROS 2 数据并转为 Rerun 记录 |
| 多源预处理示例 | https://rerun.io/examples/robotics/robot_data_preprocessing | 合并 MCAP、URDF 与外部偏移数据，处理关节变换 |
| 训练数据示例 | https://rerun.io/examples/robotics/rerun_export | 查询记录并导出为 LeRobot 数据集 |
| 技术博客 | https://rerun.io/blog/data-layer-for-robot-learning | 介绍机器人学习数据整理与训练方向 |

## 对 wiki 的映射

- [Rerun 实体页](../../wiki/entities/rerun-io.md)
- [官方仓库归档](../repos/rerun-io.md)

## 使用说明

Rerun 是机器人数据工具，不是 ROS 的替代品。离线日志可通过 MCAP 等格式导入；实时 ROS 2 使用 SDK 节点或桥接示例转发所需 topic。具体消息类型、版本兼容性和数据规模限制应以当前文档与仓库版本为准。
