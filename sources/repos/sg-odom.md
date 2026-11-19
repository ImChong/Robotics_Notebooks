# SG-ODOM 官方开源仓库与源码执行链核对

- **类型：** repo
- **代码：** https://github.com/ori-drs/SG-ODOM
- **论文：** https://arxiv.org/abs/2611.20710
- **许可证：** BSD-3-Clause；`msgs/` 第三方消息定义各自另有许可，详见仓库 README
- **核对日期：** 2026-11-19
- **核对范围：** 核对仓库 README 的依赖、安装、ROS 2 bag 回放、输入输出和三平台配置；阅读 `sg_odom/launch/sg_odom.launch.xml`、`sg_odom/CMakeLists.txt`、`sg_odom/src/LeggedEstimatorRos2Node.cpp`（`setupRosInterfaces`、`handleG1LowState`、事件入队/有序出队、`VariantRunner::processImu/processContact`）、`StabilityGatedContactClassifier.cpp`、`UnitreeG1Adapter.cpp`、`AnymalContactAdapter.cpp`、`SpotContactAdapter.cpp`、`include/gtsam/navigation/LeggedEstimator.h` 和后端实现相关代码段。未执行 Docker/colcon、未下载 bag 和模型。
- **通读：** 已完成
- **wiki：** [SG-ODOM 论文实体](../../wiki/entities/paper-sg-odom-2611-20710.md)

## 开源结论与复现入口

**已开源可辨认的 ROS 2 运行实现**，不是仅有 README 的占位仓库。官方测试条件 Ubuntu 24.04、ROS 2 Jazzy、GTSAM `develop` 源码构建（README 报告 4.3a1 开发版本，不是稳定 release）。

```bash
docker compose build
docker compose run --rm gtsam_legged_estimator
colcon build
source install/setup.bash
# 独立终端启动数据回放
ros2 bag play /path/to/bag_sequence
# 默认 Spot；也可以指定 anymal 或 g1
ros2 launch gtsam_legged_replay_example sg_odom.launch.xml robot_type:=g1
```

- **入口：** `sg_odom/launch/sg_odom.launch.xml` → ROS 2 可执行文件 `LeggedEstimatorRos2Node`（`sg_odom/src/LeggedEstimatorRos2Node.cpp`）。
- **传感器：** Spot 的 `/imu`、足端状态或 `/joint_states`；ANYmal 的 `/anymal/imu` 与 `/anymal/state_estimator/anymal_state`；G1 的 `/lowstate`（含 IMU 与 `tau_est` 等关节信息）。
- **前端：** 机器人各自 `SpotContactAdapter`、`AnymalContactAdapter`、`UnitreeG1Adapter`，提供足端 FK、接触事件；ANYmal/Spot 可在厂家 contact state 与关节力矩 GRF 模式间选择。
- **事件队列：** `LeggedEstimatorRos2Node::enqueueEvent` → `flushReadyEvents` → `handleOrderedEvent` → `dispatch` → `VariantRunner::processEvent`。
- **IMU：** `VariantRunner::processImu` → `advanceTo` → `estimator_->predict`，继续传播，不要求在每个关节采样时修正。
- **接触：** `VariantRunner::processContact` → `hasTouchdown(activeContacts)` 或超过 `maxDeadReckoningSeconds` → `estimator_->processContacts`，未触发的接触包仅被评估并跳过后端校正。输出里程计/路径/RViz marker 与 CSV/TUM 轨迹。
- **可选后端：** `estimator.variants` 只能选择 `invariant_ekf` 或 `fixed_lag_combined_bias` 中恰好一种；见 `sg_odom/config/{spot,anymal,g1}_legged_estimator.yaml`。
- **平台实现差异（重点）：** `StabilityGatedContactClassifier::classifyWindowedDwell` 用于 ANYmal/Spot 的力变化窗口与持续时段接触门控；**G1** 的 `UnitreeG1Adapter` 明确设置 `Mode::kFilteredThreshold`，经滤波竖直力和 on/off 阈值滞回决定接触（`classifyFilteredThreshold`），**不执行相同的力变化窗口 + dwell 逻辑**。不可将论文中统一的公式实现细节无区别套到 G1 上。
- **G1 时间戳：** `LowState` 无 ROS Header，节点以 ROS 接收时刻给从同包转换的 IMU、接触事件统一时间；不同录包/实时硬件时钟的同步仍须另验。

## 使用边界

仓库提供 rosbag 回放节点和可构建源码，不等于已在本次工作中成功运行；公开 README 未承诺随仓库提供全部论文基准 ROS bags。G1 `tau_est` → GRF 需验证力矩估算误差、接触高度和机体模型，算法自身没有绝对位置/yaw 全局可观性。
