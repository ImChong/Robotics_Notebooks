# ROS 2 官方文档（Open Robotics / ROS 2 社区）

> 来源归档

- **标题：** ROS 2 Documentation (Humble LTS)
- **类型：** site（官方文档）
- **来源：** Open Robotics / ROS 2 Technical Steering Committee
- **链接：** https://docs.ros.org/en/humble/
- **入库日期：** 2026-06-13
- **一句话说明：** ROS 2 权威概念、安装、通信原语、工具链与生态包索引；Humble 为当前工业与科研常用的 LTS 发行版。

## 为什么值得保留

- **事实标准中间件**：Nav2、MoveIt 2、ros2_control、Autoware、Isaac ROS 等上层栈均以 ROS 2 为集成基座。
- **一手架构定义**：DDS 通信、QoS、生命周期节点、launch 系统等核心机制以官方文档为准。
- 与本仓库 [sim2real.md](../sim2real.md)、[ros2-basics](../../wiki/concepts/ros2-basics.md)、[ros2-vs-lcm](../../wiki/comparisons/ros2-vs-lcm.md) 形成「概念—选型—部署」闭环。

## 核心概念摘录（摘自官方 Concepts）

| 概念 | 说明 |
|------|------|
| **Nodes** | 执行计算的独立进程；通过话题/服务/动作互联 |
| **Topics** | 发布/订阅异步数据流；底层由 DDS 分发 |
| **Services** | 请求/响应同步 RPC |
| **Actions** | 长时任务：目标、反馈、可取消 |
| **Parameters** | 节点级运行时配置 |
| **QoS** | 可靠性、历史深度、deadline 等策略；同一 topic 发布者与订阅者须兼容 |
| **DDS** | Data Distribution Service；ROS 2 默认中间件抽象层，去中心化、无 Master |
| **tf2** | 坐标系树与时间戳变换 |
| **Lifecycle** | 受管节点状态机（unconfigured → inactive → active） |
| **Launch** | Python/XML 描述多节点启动顺序与参数 |

## 配套一手资料

| 资源 | 链接 | 用途 |
|------|------|------|
| **ros2_control** | https://control.ros.org/humble/ | 硬件抽象层（`hardware_interface`）、控制器管理器、与 Gazebo/MuJoCo 仿真对接 |
| **Navigation2** | https://navigation.ros.org/ | 移动机器人导航参考栈 |
| **MoveIt 2** | https://moveit.picknik.ai/ | 机械臂运动规划与碰撞检测（详见 [MoveIt 2 实体页](../../wiki/entities/moveit2.md)） |
| **ROS 2 Design** | https://design.ros2.org/ | 架构决策记录（为何选 DDS、为何弃用 Master 等） |

## 与本仓库现有资料的关系

- 部署实践索引：[sim2real.md](../sim2real.md)（Humble、ros2_control、ATOM01 等）
- 概念页：[ros2-basics](../../wiki/concepts/ros2-basics.md)
- 选型对比：[ros2-vs-lcm](../../wiki/comparisons/ros2-vs-lcm.md)
- 实体案例：[navigation2](../../wiki/entities/navigation2.md)、[moveit2](../../wiki/entities/moveit2.md)、[booster-robocup-demo](../../wiki/entities/booster-robocup-demo.md)、[autoware](../../wiki/entities/autoware.md)
- 技术栈模块：[tech-map/modules/system/ros2.md](../../tech-map/modules/system/ros2.md)

## 对 wiki 的映射

- 已沉淀：[ros2-basics](../../wiki/concepts/ros2-basics.md)、[ros2-vs-lcm](../../wiki/comparisons/ros2-vs-lcm.md)
- 技术地图入口：本页支撑 [tech-node-system-ros2](../../tech-map/modules/system/ros2.md) 模块正文
