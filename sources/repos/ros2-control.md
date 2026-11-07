# ros2_control

> 来源归档（官方仓库 + control.ros.org 文档；首次由 Humanoid Motion Intelligence 开源项目主表引入，2026-11-07 改为按官方文档与源码核对）

- **标题：** ros2_control
- **类型：** repo
- **技术路线分组（据主表）：** 工程与实机部署
- **链接：** https://github.com/ros-controls/ros2_control ；文档：https://control.ros.org/ （rolling：https://control.ros.org/rolling/）
- **入库日期：** 2026-07-30；**复核日期：** 2026-11-07
- **版本：** `master` 分支（`controller_manager/package.xml` 版本 6.13.0）；文档为 rolling（页面标注 2026-11），最新稳定版 Lyrical；仍维护 Kilted / Jazzy / Humble 分支
- **许可证：** Apache-2.0（`LICENSE`、`package.xml`）
- **开源状态：** 已开源
- **一句话说明：** ROS 2 的实时控制框架：硬件接口（Actuator / Sensor / System 组件）负责读写状态与命令，Controller Manager 管理控制器生命周期并以 `read → update → write` 的实时循环驱动；同一控制器因此可以在 mock 组件、仿真器与真实设备间切换。
- **核对范围：** 仓库 `README.md`；`controller_manager/src/ros2_control_node.cpp` 全文（实时循环、SCHED_FIFO、锁内存、CPU 亲和）；control.ros.org rolling 文档中的 Getting Started（架构、Controller Manager、Resource Manager、控制器、硬件组件、URDF `<ros2_control>` 标签）、Controller Manager 用户文档（参数、诊断、确定性）、Writing a Hardware Component、Mock Components、Asynchronous Components、Running Controllers Asynchronously、Different update rates、Hardware interface types、Lifecycle of a Hardware Component、Different Clocks、Handling of Errors During read() and write()、Joint Limiting、Controller Chaining、ros2_controllers 控制器总表与 joint_trajectory_controller 的接口类型段落、Simulator Integrations 列表。**未读**：`controller_manager.cpp` 等大部分源码实现、`ros2_controllers` 各控制器的参数细节与源码、设计文档（design.ros2.org/ros-controls 的 Hardware Access / Chaining 设计文档）、各仿真器集成页内容、ros2_control_demos。
- **通读：** 已完成
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 是 → [`wiki/entities/ros2-control.md`](../../wiki/entities/ros2-control.md)

## 核对摘录

### 架构（Getting Started）

- **Controller Manager（CM）**：连接控制器侧与硬件抽象侧，并通过 ROS 服务向用户暴露入口；管理（加载/激活/停用/卸载）控制器及其所需接口，经 Resource Manager 访问硬件组件的接口；**匹配「需要的接口」与「提供的接口」，有访问冲突则报错**；控制循环由 CM 的 `update()` 管理：读硬件状态 → 更新所有活动控制器的输出 → 把结果写给硬件。
- **Resource Manager（RM）**：用 pluginlib 加载硬件组件，管理其生命周期与状态/命令接口；`read()` / `write()` 负责与硬件通信。
- **控制器**：派生自 `ControllerInterface`，以 pluginlib 插件导出，生命周期基于 `LifecycleNode`；每个周期调用 `update()`。
- **硬件组件三种类型：** `System`（复杂多自由度硬件，只有一个逻辑通信通道，如 KUKA-RSI）、`Actuator`（单自由度，如电机/阀；也可用于模块化多自由度机器人，例如每个电机独立 CAN 通信）、`Sensor`（只读，对应关节或连杆，如编码器、力矩传感器）。
- **URDF 里的 `<ros2_control>` 标签**描述硬件：`<hardware><plugin>…</plugin>`、`<joint>` 下的 `<command_interface>` / `<state_interface>`（可带 min/max 参数）、`<gpio>` 等。

### 实时循环（`ros2_control_node.cpp`）

- 主线程循环按 `cm->read(...)` → `cm->update(...)` → `cm->write(...)` 的顺序执行，然后睡到周期末（支持 sim time、阻塞式硬件同步、普通周期三种睡法）。
- 默认调用 `realtime_tools::configure_sched_fifo(thread_priority)`，**默认优先级 50**（`kSchedPriority`，源码注释称取 RT 优先级中点以便灵活）；检测到实时内核时默认 `lock_memory`；支持 `cpu_affinity` 参数绑核；`overruns.manage` 默认开启；设置失败只告警、不退出。
- 文档（Determinism）：用户默认无权设置高实时优先级，需建 `realtime` 用户组并在 `/etc/security/limits.conf` 配 `rtprio 99`、`memlock unlimited`；Docker 里需传 capability 才能设线程优先级与锁内存。

### Controller Manager 参数（文档）

- `update_rate`（只读，默认 **100** Hz）：实时更新循环频率；`enforce_command_limits`（默认 true，越限命令被钳位）；`handle_exceptions`（默认 true，捕获控制器/硬件组件抛出的异常）；`<controller_name>.fallback_controllers`（受控控制器 update 返回 ERROR 时自动激活的后备控制器，文档警告：应先在仿真中测试后备策略）；`defaults.switch_controller.strictness`（strict / best_effort）；`defaults.deactivate_controllers_on_hardware_self_deactivate`（默认 true，出于安全不建议改 false）；`hardware_components_initial_state`；以及 CM 周期、控制器与硬件组件的周期/执行时间诊断阈值（均值误差、标准差，warn / error 两级）。

### 硬件组件与异步

- 写硬件组件：继承 `Actuator/Sensor/SystemInterface`，实现生命周期回调 `on_init / on_configure / on_activate / on_deactivate / on_cleanup / on_shutdown / on_error` 与 `read()` / `write()`；状态/命令接口由框架根据 `<ros2_control>` 标签自动导出；多控制模式的硬件实现 `prepare_command_mode_switch` / `perform_command_mode_switch`。
- **异步硬件组件：** URDF 中 `is_async="true"` 可让组件在单独线程中运行，`<async>` 属性包括 `thread_priority`（默认 50）、`affinity`、`scheduling_policy`（`synchronized` 默认 / `detached`）、`print_warnings`；适用于读取很慢、会拖垮 CM 周期的传感器。ros2_controllers 的 `is_async` 与 `update_rate` 是所有控制器的通用参数。
- **错误处理：** `read()` / `write()` 返回 `ERROR` 会触发 `on_error`；成功则回到 UNCONFIGURED，**出现 ERROR/FAILURE 则进入 FINALIZED 且不可恢复，只能重新加载整个插件，目前 CM 没有对应服务**。
- **Mock Components**（`mock_components/GenericSystem`）：理想行为，把命令镜像到状态，用于不接硬件时测试控制器、broadcaster、launch 文件甚至 MoveIt 集成。
- **关节限位：** 全局参数 `enforce_command_limits`，也可在 URDF 的 `<limits enable="false"/>` 针对整个关节或单个接口关闭（限位算法章节在文档中仍为 tba）。
- **控制器链（Controller Chaining）：** 串行链式/级联控制，后级控制器导出引用接口，前级控制器认领；链上任一控制器停用，其前级也须停用。

### ros2_controllers

- 控制器总表：Admittance、Forward Command、GPIO Command、Joint Trajectory、Motion Primitive、Parallel Gripper、PID 以及差速/麦轮/全向/Ackermann 等移动底盘控制器；Broadcaster（Joint State、IMU、Force Torque 等）本质仍是用同一控制器接口的控制器；Chained Filter Controller。
- **joint_trajectory_controller：** 在关节空间插值执行轨迹（航点含位置，可含速度、加速度）；支持命令接口组合 position / position+velocity / position+velocity+acceleration / velocity / effort / position+effort；velocity 或 effort 接口时对位置+速度误差用 PID 映射为命令，effort 还加轨迹前馈。

### 仿真集成

- 文档列出的集成：Gazebo、MuJoCo（ros-controls 托管）、Isaac Sim、Webots、MuJoCo（FZI）、Algoryx AGX Dynamic。


### 异步控制器、多速率与时钟（2026-11-07 补读）

- **异步控制器**（Running Controllers Asynchronously）：当控制器 `update()` 含阻塞调用或耗时超过 CM 周期时，应让它在专用线程中运行；对控制器实现透明，只需把 `is_async` 设为 true；`update_rate` 可设为低于 CM 的速率（文档示例：CM 100 Hz，异步控制器 20 Hz）；线程参数在 `async_parameters` 下（`thread_priority`、`cpu_affinity`，线程用 FIFO 调度）。**调度行为：** CM 充当调度器触发异步 `update()`；若上一次 `update()` 尚未结束，则**复用上一次结果**并打印 "missed N update cycles" 告警，文档建议频繁出现时降低 `update_rate`；未处理异常则停用该控制器。依据：同步情形下全部控制器 `update()` 与硬件 `read()/write()` 的总耗时必须小于 CM 周期（100 Hz 即 10 ms）。
- **硬件组件多速率**：`<ros2_control>` 标签上的 `rw_rate` 属性（如 500、200）可让不同硬件组件以不同于 CM 的频率读写，前提是 CM 频率更高。
- **接口数据类型**：命令/状态接口默认 `double`（初值 NaN），可用 `data_type` 指定 float32、bool、(u)int8/16/32；`<gpio>` 描述不属于关节/传感器的端口，没有通用发布器，需要自写 GPIO 控制器；**Hardware Groups** 用于错误传播（默认某个执行器的错误只隔离在其自身组件内）。
- **硬件组件生命周期**：UNCONFIGURED（只初始化）→ INACTIVE（已通信，可读状态；System/Actuator 的命令接口不可用，但文档称**目前所有命令接口都可用且会被写出**，安全关键接口尚在计划中）→ ACTIVE（上电、抱闸释放、命令下发）→ FINALIZED。
- **时钟：** 实时 read/update/write 循环主要用 `RCL_STEADY_TIME`（单调）；硬件组件 `read/write` 的 time 参数是 steady time，控制器 `update` 的 time 是 ROS time（便于校验收到的命令、给发布消息打时间戳）；仿真时用 ROS time 触发。ros2_control_node 现在能检测因系统时间跳变或执行时间过长引起的周期超限。

## 为什么值得保留

主表将该项列为人形运动智能六条路线之一的工程/研究入口；本库为其建立独立详情节点，便于选型与交叉引用，而不镜像主表全文。

## 对 wiki 的映射

- [ros2_control](../../wiki/entities/ros2-control.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
