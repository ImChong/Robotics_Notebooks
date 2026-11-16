# GitHub 仓库：ZJU-FAST-Lab/ego-planner-v2（Swarm Playground）

- **仓库：** https://github.com/ZJU-FAST-Lab/ego-planner-v2
- **关联项目：** Swarm of Micro Flying Robots in the Wild（EGO-Planner-v2）
- **开源状态：** 已开源（完整 ROS1 C++ 规划器节点、L-BFGS + MINCO 轨迹优化器、UDP 集群通信桥 `swarm_bridge` 与多机仿真启动栈）
- **核对范围：** 已通读官方仓库 `ego-planner-v2` 中的 `README.md`、`swarm-playground/main_ws/src/planner/plan_manage/src/{ego_planner_node.cpp, ego_replan_fsm.cpp, planner_manager.cpp, traj_server.cpp}`、`traj_opt/src/poly_traj_optimizer.cpp` 及 launch 文件 `swarm.launch`、`include/run_in_sim.xml`、`include/advanced_param.xml`
- **通读：** 已完成

## 代码结构与核心模块

1. **主入口与有限状态机（`ego_planner_node.cpp` / `ego_replan_fsm.cpp`）**
   - `ego_planner_node` 实例化 `EGOReplanFSM` 并执行 `init(nh)` 与 `ros::spin()`。
   - `EGOReplanFSM::init` 初始化 `PlanningVisualization`、`EGOPlannerManager::initPlanModules`，注册 **100 Hz** 状态机定时器 `exec_timer_`（`execFSMCallback`）与 **20 Hz** 安全检查定时器 `safety_timer_`（`checkCollisionCallback`）。
   - 订阅 `odom_world`、`mandatory_stop` 与集群广播话题 `planning/broadcast_traj_recv`（`RecvBroadcastMINCOTrajCallback`）；发布机内控制轨迹 `planning/trajectory`（`traj_utils::PolyTraj`）、心跳 `planning/heartbeat` 与机间广播 `planning/broadcast_traj_send`（`traj_utils::MINCOTraj`）。
   - 状态流转覆盖 `INIT -> WAIT_TARGET -> SEQUENTIAL_START / GEN_NEW_TRAJ -> EXEC_TRAJ <-> REPLAN_TRAJ / EMERGENCY_STOP`。

2. **规划调度与 MINCO L-BFGS 轨迹优化（`planner_manager.cpp` / `poly_traj_optimizer.cpp`）**
   - `EGOPlannerManager::reboundReplan` 先基于 `MinJerkOpt` 初始化或续接初态并调用 `finelyCheckAndSetConstraintPoints` 建立静态障碍碰撞排斥方向，再支持多拓扑并行路径搜索（`distinctiveTrajs`）或单条 `optimizeTrajectory`。
   - `PolyTrajOptimizer::optimizeTrajectory` 将实时间 $T$ 映射为无约束虚拟时间变量（`RealT2VirtualT`），调用 `lbfgs::lbfgs_optimize`（`mem_size=16`、`max_iterations=200`）优化 `costFunctionCallback`（内部计算平滑梯度 `initAndGetSmoothnessGradCost2PT` 与环境障碍/机间互避/可行性代价 `addPVAJGradCost2CT`，经 `earlyExitCallback` 触发多达 20 次 rebound 与 3 次异常重启）。

3. **轨迹伺服与多机仿真栈（`traj_server.cpp` / `swarm.launch`）**
   - `traj_server` 接收五次分段多项式 `PolyTraj` 与心跳，以 **100 Hz**（`0.01s`）`cmdCallback` 计算 $p, v, a, j$ 与前向偏航 `calculate_yaw` 并输出 `quadrotor_msgs::PositionCommand`；若心跳超时 `> 0.5s` 自动停机保护。
   - `swarm.launch` 通过 `swarm_bridge`（`bridge_udp.launch`）完成各机间 UDP 广播，并在单机侧加载 `run_in_sim.xml` + `advanced_param.xml`（默认 `planning_horizon=7.5m`、`max_vel=1.5`、`max_acc=6.0`、`max_jer=20.0`）。
