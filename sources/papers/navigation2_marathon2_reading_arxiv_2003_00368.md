# The Marathon 2: A Navigation System 通读（Navigation2 原始论文）

> 来源归档（原文通读 · 支撑 `wiki/overview/navigation-slam-autonomy-stack.md`）

- **标题：** The Marathon 2: A Navigation System
- **作者：** Steve Macenski, Francisco Martín, Ruffin White, Jonatan Ginés Clavero
- **类型：** paper / navigation / ros2 / behavior-tree
- **原文：** <https://arxiv.org/abs/2003.00368>（arXiv PDF v2，2020-07-14，通过 PDF 获取全文）
- **一句话说明：** 提出 Navigation2：以行为树编排规划/控制/恢复，任务服务器 + 运行时插件，建立在 ROS 2 生命周期节点之上，并用两台差速机器人在校园跑完 37.4 英里长时实验。
- **核对范围：** 通读摘要、§I–§VII 全文（含相关工作、设计、实现、实验与分析、局限、结论）及参考文献列表；未阅读其引用的 STVL、TEB、SLAM Toolbox 等原始论文，也未核对其后 Nav2 的版本演进（Smac、MPPI、RPP 等不在本文范围）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **设计目标（§III）：** 支持差速、全向、阿克曼、腿式等多种机器人；强调可靠（ROS 2 / DDS、生命周期 Managed Node 提供确定性启动与关闭）、模块化与运行时可配置。
- **架构（§III-B，图 2）：** BT Navigator 用行为树（XML，运行时加载，基于 BehaviorTree.CPP）编排；BT 节点通过 ROS 2 action 调用 planner / controller / recovery 等任务服务器，每个服务器承载环境模型 + 运行时选择的算法插件，可跨进程/CPU 异步运行。
- **行为树与恢复（§IV-A/B，图 3）：** 全局规划 1 Hz；规划失败则清代价地图，再依次升级为 Clear Costmap → Spin → Wait，恢复行为按「保守到激进」排序。
- **感知（§IV-C）：** 分层代价地图（Static / Inflation / Voxel / STVL 等）；STVL 维护带时间衰减的稀疏 3D 体素，比射线投射更适合多/远/稀疏传感器。
- **规划与控制（§IV-D，表 I）：** NavFn（A* 展开，假设 2D 全向质点）、DWB（可配置 DWA）、TEB（时间最优非线性 MPC）；实验用 NavFn-A* + TEB，DWB 因动态场景表现未使用。
- **状态估计（§IV-E）：** 遵循 ROS TF 标准；Robot Localization（EKF/UKF）融合里程计；SLAM Toolbox 用于建静态图，AMCL 做实验期定位。
- **QA（§IV-G）：** Gazebo 仿真集成测试进入 CI。
- **实验（§V，表 III）：** Tiago 与 RB-1 两台差速机器人，限速 0.45 m/s，共 22.8 小时、37.4 英里、168 次恢复（4.3 次/英里）、平均速度 0.37 m/s、0 碰撞、0 急停；被动协助总计不足 1 分钟。恢复主因：拥挤时无法规划；长走廊重复特征 + 人流导致 AMCL 置信度下降（Spin 帮助重定位）。
- **局限（§VI）：** A* 规划不为非圆形非完整约束机器人（如车型）给出可行路径；实验未用显式动态障碍检测与预测；动态障碍跟踪/规划留待后续。

## 对 wiki 的映射

- [`wiki/overview/navigation-slam-autonomy-stack.md`](../../wiki/overview/navigation-slam-autonomy-stack.md)
- [`wiki/entities/navigation2.md`](../../wiki/entities/navigation2.md)
