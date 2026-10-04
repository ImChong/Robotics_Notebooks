# Rerun

> 官方仓库归档

- **标题：** Rerun
- **类型：** repo / 多模态机器人数据可视化与处理工具
- **开发方：** Rerun Technologies
- **链接：** https://github.com/rerun-io/rerun
- **官方文档：** https://rerun.io/docs/
- **入库日期：** 2026-07-30
- **最近复核：** 2026-10-04
- **开源状态：** **核心 SDK 与 Viewer 已开源**；Rerun Hub 是面向团队与规模化数据管理的商业产品。
- **许可证：** 仓库包含 Apache-2.0、MIT 等许可证；具体组件以仓库 LICENSE 文件为准。
- **一句话说明：** 面向机器人与 Physical AI 的多模态数据层：记录、导入、同步可视化、查询并将机器人记录接入训练数据流程。
- **官方站点归档：** [Rerun 官方站点与文档](../sites/rerun-io.md)
- **沉淀到 wiki：** 是 → [Rerun 实体页](../../wiki/entities/rerun-io.md)

## 项目概览

Rerun 提供 Python、Rust、C++ SDK 和 Viewer。SDK 可在应用中记录图像、点云、变换、关节状态、标量与视频等数据；数据可以实时发送给 Viewer，也可以保存为 Rerun Data（.rrd）文件。其文档还覆盖 MCAP 导入、数据查询、预处理和面向机器人学习的数据加载流程。

## 机器人数据入口

- **程序内记录：** 用 SDK 给多频率数据关联时间轴与实体路径，再连接 Viewer 或保存为 .rrd。
- **已有记录：** Viewer 可打开 .rrd、MCAP 等格式；MCAP 导入器可解析常见 ROS 2 / Foxglove 消息，并通过反射读取其他受支持消息。
- **ROS 2 实时数据：** 官方示例展示 ROS 2 节点订阅图像、点云、激光扫描、TF、里程计与 URDF，并转成 Rerun 数据。
- **机器人模型：** URDF 导入器加载网格与关节框架；后续通过带父子 frame 的变换更新关节状态。
- **查询与训练：** DataFrame / SQL 查询和 Dataloader 将同一套记录用于筛选、转换、训练数据整理。

## 官方架构与使用材料

- [README 与安装入口](https://github.com/rerun-io/rerun)
- [项目架构说明](https://github.com/rerun-io/rerun/blob/main/ARCHITECTURE.md)
- [官方文档：Rerun 的工作方式](https://rerun.io/docs/concepts/how-does-rerun-work)
- [官方示例：ROS 2 节点](https://rerun.io/examples/robotics/ros_node)
- [官方文档：MCAP 导入](https://rerun.io/docs/howto/logging-and-ingestion/mcap)
- [官方文档：URDF 模型导入](https://rerun.io/docs/howto/logging-and-ingestion/urdf)
- [官方示例：机器人数据预处理](https://rerun.io/examples/robotics/robot_data_preprocessing)
- [官方博客：机器人学习数据层](https://rerun.io/blog/data-layer-for-robot-learning)

## 对 wiki 的映射

- [Rerun 实体页](../../wiki/entities/rerun-io.md)
- [官方站点与文档归档](../sites/rerun-io.md)

## 复核备注

截至 2026-10-04，官方 README 将项目描述为面向多模态机器人数据的记录、查询、可视化与训练工具，并注明项目仍在积极开发、API 可能发生破坏性变更。大规模实体与超大点云场景应按具体数据验证 Viewer 性能。
