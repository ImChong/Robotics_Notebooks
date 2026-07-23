# TARE Planner（tare_planner）

> 来源归档

- **标题：** TARE Exploration Planner for Ground Vehicles
- **类型：** repo
- **链接：** https://github.com/caochao39/tare_planner
- **项目页：** https://www.cmu-exploration.com/tare-planner
- **机构：** Carnegie Mellon University Robotics Institute（CMU）
- **Stars：** ~687（核查日 2026-07）
- **入库日期：** 2026-07-23
- **开源状态：** **已开源**（ROS Melodic/Noetic 分支）
- **一句话说明：** CMU 分层自主探索规划器：局部密表示细路径 + 全局稀表示粗路径，两层均以 TSP 近似最优游览未知空间；DARPA SubT 实战验证。
- **沉淀到 wiki：** [tare-planner](../../wiki/entities/tare-planner.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md)

---

## 核心定位

**TARE**（Technologies for Autonomous Robot Exploration）用 **表示粒度分层** 换算力：近场高分辨率规划细节，远场低分辨率保持全局态势，避免对整张稠密地图做全局细规划。

代表论文：

- Cao et al., *TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments*, RSS 2021（Best Paper / Best System Paper）
- Cao et al., *Representation Granularity Enables Time-Efficient Autonomous Exploration…*, Science Robotics 2023

与 [FAR Planner](far_planner.md) 同属 [CMU Exploration 开发环境](../sites/cmu-exploration.md) 规划栈：TARE 做 **探索目标生成**，FAR 做 **到目标的快速可尝试路由**。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 方法实体 | [tare-planner](../../wiki/entities/tare-planner.md) |
| 自主探索任务 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 课程映射 | [humanoid-system-curriculum](../../wiki/entities/humanoid-system-curriculum.md) Ch5 |
| 对照路由规划 | [far-planner](../../wiki/entities/far-planner.md) |
