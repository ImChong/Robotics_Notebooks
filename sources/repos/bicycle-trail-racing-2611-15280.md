# Autonomous Trail Racing with a Bicycle Robot 官方源码核对

- **类型：** 官方源码仓
- **代码：** https://github.com/rai-opensource/sem_map
- **论文：** https://arxiv.org/abs/2611.15280
- **公开运行入口：** `ros2 launch sem_map sem_map_camera.launch.py image_topic:=/camera/color/image_raw`
- **相关模块：** `ComposableNodeContainer 与 SemMap`
- **运行产物：** 语义分割、可通行性和可选高程图
- **核对范围：** 官方 README、该仓库公开入口脚本与关键模块的静态核查；外部数据和模型依赖未执行。
- **通读：** 已完成
- **wiki：** [论文独立详情](../../wiki/entities/paper-bicycle-trail-racing-2611-15280.md)
- **原文：** [论文来源](../papers/bicycle-trail-racing-2611-15280.md)

高草和植被可能制造虚假障碍，岔路选错与轮胎失去抓地力会影响安全；公开 SemMap 感知仓不是整个骑行控制栈。
