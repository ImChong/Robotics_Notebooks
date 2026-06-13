# ROS 2（系统集成层）

**ROS 2**（Robot Operating System 2）是机器人全栈里**系统集成与多进程编排**的事实标准中间件：它把感知、规划、控制、标定与调试工具接到同一套通信与工具链上，是 [Sim2Real](../../../wiki/concepts/sim2real.md) 与真机部署阶段的常见「胶水层」。

> 本页是技术栈模块入口；概念细节见 [ROS 2 基础](../../../wiki/concepts/ros2-basics.md)，底层高频运控选型见 [ROS 2 vs LCM](../../../wiki/comparisons/ros2-vs-lcm.md)。

## 这个模块解决什么问题

- **多传感器、多算法、多进程如何可靠互联**：雷达驱动、SLAM、导航、机械臂规划、状态机决策等组件常以独立节点运行，ROS 2 提供统一的 pub/sub、服务、动作与参数接口。
- **坐标系与时间如何对齐**：`tf2` 管理 `map` → `odom` → `base_link` → 各传感器外参；与 [时钟同步算法](../../../wiki/concepts/clock-synchronization-algorithms.md) 配合可缓解多源数据时间戳漂移。
- **从仿真到真机的工程落地**：`ros2_control` 抽象硬件接口；launch 文件描述启动拓扑；`rosbag2` 录制回放便于调试与回归。

在本仓库当前主线（人形/腿式运控 + 学习 + sim2real）里，ROS 2 **通常承担中高层**（10–100 Hz 感知与规划），而 **500–1000 Hz 关节闭环** 更常走 LCM / 共享内存 / EtherCAT——见 [实时运控中间件配置指南](../../../wiki/queries/real-time-control-middleware-guide.md)。

## 在全栈中的位置

```text
控制 / 学习策略（MPC、WBC、RL policy）
  ↓
状态估计 · 系统辨识 · Sim2Real
  ↓
ROS 2 系统集成层  ← 本模块
  ├─ 感知节点（相机、LiDAR、IMU 驱动）
  ├─ 规划与决策（Nav2、行为树、VLA 接口）
  ├─ 硬件抽象（ros2_control、底盘/机械臂驱动）
  └─ 调试与可视化（RViz、Foxglove、rosbag2）
  ↓
真机驱动 · 安全监控 · OTA 部署
```

与相邻模块的关系：

| 上游 | 本模块 | 下游 |
|------|--------|------|
| [仿真](../../../tech-map/modules/system/simulation.md) 中验证的策略与场景 | 把算法包装成可部署的节点图 | [部署](../../../tech-map/modules/system/deployment.md) 阶段的频率隔离、安全与上线 |
| [Sim2Real](../../../wiki/concepts/sim2real.md) 训练分布与域随机化 | 桥接仿真话题与真机驱动 | 具体平台仓库（如 ATOM01、Pupper v3） |

## 和「ROS 2 基础概念页」的区别

| 页面 | 职责 |
|------|------|
| **本页（技术栈模块）** | ROS 2 在全栈集成层的角色、分层用法、本仓库案例入口 |
| **[ROS 2 基础](../../../wiki/concepts/ros2-basics.md)** | DDS、节点/话题/服务/动作等概念定义 |
| **[ROS 2 vs LCM](../../../wiki/comparisons/ros2-vs-lcm.md)** | 中高层生态 vs 底层高频运控的选型与混合架构 |

## 核心内容

### 1. 通信与中间件（DDS）

ROS 2 相对 ROS 1 的最大变化是采用 **DDS** 作为默认中间件实现：

- **去中心化**：无 `roscore` Master，单节点崩溃不拖垮全网。
- **QoS 可配**：可靠性、历史深度、deadline 等；同一 topic 两端策略须兼容。
- **代价**：协议栈较重；**不宜**承载 500 Hz 以上的硬实时关节闭环（见对比页）。

### 2. 硬件与控制接口（ros2_control）

[`ros2_control`](https://control.ros.org/humble/) 提供 **Resource Manager + Controller Manager**：

- `hardware_interface` 统一关节状态读回与命令写出；
- 控制器插件（位置/速度/力矩、轨迹跟踪）与仿真后端（Gazebo、Isaac）可替换；
- 是人形/机械臂从「算法输出」到「电机命令」的常见工程路径之一。

本仓库 [sim2real 部署索引](../../../sources/sim2real.md) 将 Humble + ros2_control + [ATOM01 部署](../../../sources/repos/atom01_deploy.md) 列为典型组合。

### 3. 生态栈（按任务选）

| 任务域 | 代表项目 | 本仓库入口 |
|--------|----------|------------|
| 移动导航 | Nav2 | [Navigation2](../../../wiki/entities/navigation2.md)、[导航·SLAM 栈总览](../../../wiki/overview/navigation-slam-autonomy-stack.md) |
| 自动驾驶 | Autoware Universe | [Autoware](../../../wiki/entities/autoware.md) |
| 视觉 SLAM | Isaac ROS、slam_toolbox | [Isaac ROS Visual SLAM](../../../wiki/entities/isaac-ros-visual-slam.md) |
| 群体无人机 | Crazyswarm2 | [Crazyswarm2](../../../wiki/entities/crazyswarm2.md) |
| 人形足球 demo | Booster RoboCup | [Booster RoboCup Demo](../../../wiki/entities/booster-robocup-demo.md) |
| 四足机载软件 | Pupper v3 monorepo | [sources/repos/pupperv3_monorepo.md](../../../sources/repos/pupperv3_monorepo.md) |

### 4. 推荐分层架构（混合栈）

先进系统常采用 **「ROS 2 大脑 + LCM/共享内存小脑」**：

1. **慢路径（IPC / Jetson）**：ROS 2 跑 SLAM、全局规划、VLA/行为树，10–30 Hz 输出目标位姿或足端参考。
2. **快路径（PREEMPT_RT 核心）**：MPC / RL / WBC 以 500–1000 Hz 闭环，进程内用共享内存或 **LCM** 与驱动通信。
3. **桥接节点**：将 ROS 2 低频指令转换为快路径可消费的结构体，避免 DDS 抖动进入控制环。

详见 [ROS 2 vs LCM](../../../wiki/comparisons/ros2-vs-lcm.md) 与 [实时运控中间件配置指南](../../../wiki/queries/real-time-control-middleware-guide.md)。

## 学完应该会什么

- 能说明 ROS 2 在全栈中**负责集成而非替代底层运控**。
- 能根据任务选择 Nav2 / ros2_control / 纯自定义节点等落点。
- 知道何时必须引入 LCM 或共享内存桥接，以及 Linux RT 与 CPU 隔离的基本动机。

## 最小落地任务

1. 按 [ROS 2 Humble 安装文档](https://docs.ros.org/en/humble/Installation.html) 搭好工作区，跑通 `talker`/`listener` 示例。
2. 用 `rviz2` 可视化 `tf2` 树，理解 `base_link` 与传感器外参关系。
3. 阅读一个本仓库案例 launch 图（如 [Booster RoboCup Demo](../../../wiki/entities/booster-robocup-demo.md) 或 [Navigation2](../../../wiki/entities/navigation2.md) 数据流），标出感知→规划→控制的节点边界。
4. （可选）在同一机器上对比 ROS 2 topic 与 LCM channel 的 1 kHz 往返延迟，验证「分层」必要性。

## 推荐一手资料

- [ROS 2 官方文档（Humble）](https://docs.ros.org/en/humble/) — 概念、安装、工具链（本仓库归档：[sources/sites/ros2-official-documentation.md](../../../sources/sites/ros2-official-documentation.md)）
- [ROS 2 Design](https://design.ros2.org/) — 架构决策记录
- [ros2_control 文档](https://control.ros.org/humble/) — 硬件抽象与控制器
- [Navigation2 文档](https://navigation.ros.org/) — 移动机器人导航栈

## 关联页面

- [ROS 2 基础](../../../wiki/concepts/ros2-basics.md)
- [ROS 2 vs LCM](../../../wiki/comparisons/ros2-vs-lcm.md)
- [实时运控中间件配置指南](../../../wiki/queries/real-time-control-middleware-guide.md)
- [Sim2Real](../../../wiki/concepts/sim2real.md)
- [仿真（系统集成层）](./simulation.md)
- [部署](./deployment.md)
- [模块依赖关系图](../../dependency-graph.md)

## 难度 / 优先级

- **难度**：3/5（概念清晰，但 DDS/QoS/launch 与真机驱动调试曲线陡）
- **优先级**：3/5（当前仓库主攻运控与学习主线；集成层在 sim2real 落地阶段重要性上升）
