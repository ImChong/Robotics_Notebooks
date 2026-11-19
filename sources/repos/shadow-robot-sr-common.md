# Shadow Robot `sr_common`（URDF 模型描述与 ROS 消息包）

> 来源归档

- **标题：** `shadow-robot/sr_common`
- **类型：** repo（ROS URDF 模型描述与消息定义集）
- **组织：** Shadow Robot Company
- **链接：** <https://github.com/shadow-robot/sr_common>
- **许可：** GPL / BSD（各子包分别声明，见 `package.xml`）
- **入库日期：** 2026-11-19
- **通读：** 已完成
- **核对范围：** 通读 `sr_common/README.md`、`sr_description/CMakeLists.txt`、`sr_description/package.xml`、`sr_description/robots/sr_hand.urdf.xacro`、`sr_description/hand/xacro/hand.urdf.xacro`、`sr_description/test/test_sr_description_urdf.cpp` 及 `sr_robot_msgs/`（`CMakeLists.txt` 与核心 `.msg`/`.srv` 接口定义）
- **一句话说明：** Shadow Robot 官方维护的 ROS 公共基础仓库，提供可参数化拼装的 Shadow Hand URDF/xacro 几何与传感器模型（`sr_description`）以及完整电机、EtherCAT 调试与触觉消息/服务接口（`sr_robot_msgs`），为远端可视化、MoveIt/Gazebo 仿真与真机驱动栈提供统一数据契约。
- **沉淀到 wiki：** [Shadow Hand](../../wiki/entities/shadow-hand.md)

---

## 步骤 2.5：源码开放核查

| 判定 | 边界 |
|------|------|
| **已开源（模型与接口层）** | `sr_common`（包含 `sr_description` URDF/xacro 模型、`meshes/` 网格、`sr_robot_msgs` 消息/服务/Action 定义）全部公开 |
| **真机驱动外置 / 固件闭源** | 真实硬件的 1 kHz EtherCAT 主站控制栈位于独立开源/维护仓库 `shadow_robot_ethercat`；而手掌/前臂内部 Smart Motor PIC 电机板固件与电路原理图受 **NDA 保密协议** 约束，不开源 |

---

## 核心子包解析

1. **`sr_description`（参数化运动学与仿真模型）：**
   - `robots/sr_hand.urdf.xacro` 定义顶层宏调用入口，默认参数 `hand_type="hand_e"`、`hand_version="E3M5"`、`side="right"`、`fingers="all"`、`tip_sensors="pst"`（另支持 `mid_sensors`、`prox_sensors`、`palm_sensor`、`mounting_plate`）。
   - `hand/xacro/hand.urdf.xacro` 根据 `side`（设置 `rh_`/`lh_` 链路前缀与镜像符号 `reflect=±1`）与 `hand_type`（分派 `E` / `G` / `C` 型号宏 `sr_hand_e` / `sr_hand_g` / `sr_hand_c`），按前臂 → 腕关节 → 手掌 → 5 指与远端耦合关节组装完整 URDF，并挂载 `gazebo.urdf.xacro` 传输与插件标签。
   - `test/test_sr_description_urdf.cpp` 通过遍历 `robots/*.urdf.xacro` 并自动运行 `xacro` + `check_urdf` 完成全构型静态合法性测试。

2. **`sr_robot_msgs`（统一硬件状态与控制契约）：**
   - **EtherCAT 底层与电机遥测：** `EthercatDebug.msg`（含 `sensors`、`FromMotorDataType`、`which_motors`、`which_motor_data_arrived`、`which_motor_data_had_errors`、`motor_data_packet_torque`、`motor_data_packet_misc`、`tactile`、`idle_time_us`）、`ActuatorStatistics.msg`、`JointStatistics.msg`。
   - **电机参数与控制模式切换：** `ControlType.msg`（区分 `PWM=0`、`FORCE=1`、`QUERY=-1`）、`ChangeControlType.srv`、`MotorSystemControls.msg`（在线启用/关闭齿隙补偿 `enable_backlash_compensation`、调整左右应变片校准、触发自检 `initiate_jiggling` 与写入 EEPROM）。
   - **多模态触觉消息：** `Tactile.msg`（`Int16[]` 通用触觉数组）、`ShadowPST.msg`（气压指尖）、`Biotac.msg` / `BiotacAll.msg`（BioTac 指尖兼容）、`UBI0.msg` / `MST.msg` 与接触状态 `ShadowContactStateStamped.msg`。
   - **上层规划 Action：** `Grasp.action`、`PlanGrasp.action` 及命名轨迹规划服务 `PlanNamedTrajectory.srv` / `ExecutePlannedTrajectory.srv`。

---

## 对 wiki 的映射

- 实体页：[Shadow Hand](../../wiki/entities/shadow-hand.md)
- 配套规格归档：[shadow-dexterous-hand-specs.md](../sites/shadow-dexterous-hand-specs.md)
