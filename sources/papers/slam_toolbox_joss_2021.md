# SLAM Toolbox: Open Source ROS2 Toolkit for Lifelong Mapping and Localization（JOSS 2021）

> 论文来源归档（ingest）

- **标题：** SLAM Toolbox: SLAM for the dynamic world
- **类型：** paper / open-source software / 2d-slam / ros2 / lifelong-mapping
- **DOI：** <https://doi.org/10.21105/joss.02783>
- **期刊：** Journal of Open Source Software (JOSS), 6(61), 2783（2021-05）
- **作者：** Steve Macenski, Ivona Jambrecic
- **官方仓库：** <https://github.com/SteveMacenski/slam_toolbox>
- **入库日期：** 2026-11-21
- **通读：** 已完成
- **一句话说明：** 提出面向工业生产级部署的 ROS 2 2D 激光 SLAM 工具包，基于增强版 Open Karto 前端与 Google Ceres 位姿图优化（SPA）后端，原生支持建图会话的序列化/反序列化、终身增量建图（Lifelong Mapping）、同步与异步建图切换、纯定位及 RViz 交互式闭环微调。

---

## 核心摘录（面向 wiki 编译）

### 1) 面向工业终身运行的定位与架构
- **痛点：** 早期开源 2D SLAM 多面向研究验证，面对仓库货架变更、多班次重启或超大规模场馆，缺乏图持久化、状态恢复与长时间无漂移更新机制。
- **架构：** 以 `Open Karto` 作为扫描匹配与位姿图（Pose Graph）构建骨架，将非线性最小二乘求解替换/增强为 **Google Ceres Solver** 支持的稀疏位姿调整（SPA），并通过 `ROS 2 Lifecycle` 节点管理工程化生命周期。

### 2) 四大核心运行模式
- **同步（Synchronous）与异步（Asynchronous）在线建图：** 同步模式严格按序处理每帧 `LaserScan`；异步模式以最佳努力机制丢弃积压旧帧，保障低端机载 CPU 实时性。
- **离线（Offline）与纯定位（Localization）：** 离线从 `.bag` 高保真重构图谱；纯定位基于已保存位姿图做滚动局部窗口匹配（不长期膨胀内存）。
- **终身建图（Lifelong Mapping）与序列化：** 支持把当前位姿图与原始扫描状态序列化至 `.posegraph` + `.data`，重加载后继续建图，并通过动态置信/节点衰减策略控制图规模。

### 3) 交互式工具链与 Nav2 生态集成
- 内置 RViz 插件支持人工辅助闭环约束调整与子节点拖拽，用于高对称长廊（如货架区）误匹配修复。
- 作为 **Navigation2（Nav2）** 官方推荐默认 2D 建图与 SLAM 后端，标准输出 `/map` 占据栅格与 `map -> odom` TF 变换。

---

## 对 wiki 的映射

- 实体页：[`wiki/entities/slam-toolbox.md`](../../wiki/entities/slam-toolbox.md)
- 仓储归档：[`sources/repos/slam_toolbox.md`](../repos/slam_toolbox.md)
- 选型对比：[`wiki/comparisons/lidar-slam-lio-vio-selection.md`](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)
