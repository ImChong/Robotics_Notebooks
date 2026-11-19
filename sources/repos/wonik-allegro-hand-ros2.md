# Wonik Robotics Allegro Hand ROS 2 (V5) 与 ROS 1 (V4) 驱动仓库

> 来源归档

- **标题：** `allegro_hand_ros2_v5` & `allegro_hand_ros_v4`（Wonik Robotics 官方软件栈）
- **类型：** repo（ROS 2 / ROS 1 C++ 驱动与控制器）
- **组织：** Wonik Robotics
- **仓库范围：** `allegro_hand_ros2_v5`（ROS 2 Humble 验证）与 `allegro_hand_ros_v4`（ROS 1 历史对照）
- **许可：** BSD License
- **入库日期：** 2026-11-19
- **通读：** 已完成
- **核对范围：** 通读 `allegro_hand_ros2_v5/README.md`、`src/allegro_hand_driver/`（`AllegroHandDrv.h`、`AllegroHandDrv.cpp`、`AllegroHandDef.h`、`candrv/candef.h`、`candrv/socket_can.cpp`）、`src/allegro_hand_controllers/`（`CMakeLists.txt`、`launch/allegro_hand.launch.py`、`src/allegro_node.h`、`src/allegro_node.cpp`、`src/allegro_node_grasp.cpp`）及 `allegro_hand_ros_v4/README.md` 与 `allegro_node.h`/`AllegroHandDrv.cpp` 对照
- **一句话说明：** Wonik Robotics 官方提供的 Allegro Hand V5 ROS 2（Humble）C++ 驱动、SocketCAN 通信层、`BHand` 抓取/重力补偿/PD 控制器与 Isaac Sim 桥接源码，明确区分 V5（2 ms / 500 Hz，SocketCAN `can0`）与 V4（3 ms / 333 Hz，`/dev/pcan*`）的实时控制链路。
- **沉淀到 wiki：** [Allegro Hand](../../wiki/entities/allegro-hand.md)

---

## 步骤 2.5：源码开放核查

| 判定 | 边界 |
|------|------|
| **已开源** | ROS 2 控制节点、CAN 驱动层（SocketCAN）、`BHand` 控制库与 Isaac Sim 对接代码均以 BSD 协议公开 |
| **硬件/固件边界** | 手部嵌入式微控制器固件以二进制发布（经内部 USB-C Y-modem 升级），不开源固件源码 |

---

## 代码结构与核心实现（`allegro_hand_ros2_v5`）

1. **驱动与 SocketCAN 传输（`allegro_hand_driver`）：**
   - `socket_can.cpp` 通过 Linux 标准 `socket(PF_CAN, SOCK_RAW, CAN_RAW)` 打开并绑定 `can0`（依赖内核内置 `peak_usb`，设为 `O_NONBLOCK` 非阻塞读取）。发送和接收帧时对 CAN ID 进行 2 位左/右移位编码（`msg.can_id = (id << 2)` / `*id = (msg.can_id >> 2)`）。
   - `AllegroHandDef.h` 定义 16 关节枚举 `eJOINTNAME_INDEX_0..3`、`eJOINTNAME_MIDDLE_0..3`、`eJOINTNAME_PINKY_0..3`（对应物理无名指位）与 `eJOINTNAME_THUMB_0..3`（共 `DOF_JOINTS = 16`）。
   - `candef.h` 定义底层命令常量：`ID_CMD_SYSTEM_ON` (`0x40`)、`ID_CMD_SYSTEM_OFF` (`0x41`)、`ID_CMD_SET_TORQUE_1..4` (`0x60..0x63`)、`ID_RTR_FINGER_POSE_1..4` (`0x20..0x23`)、`ID_CMD_FINGERTIP_1..4` (`0x50..0x53`) 等。
   - `AllegroHandDrv::init()` 执行缓冲清空、伺服关闭、查询手部硬件信息与序列号（自动识别左/右手 `RIGHT_HAND` 与 `HAND_TYPE_A`），随后上电伺服。注：源码中 `init()` 仍保留 `comm_period[3] = {3, 0, 0}` 的历史设定，但 `allegro_node.h` 已将控制步长切换为 `#define ALLEGRO_CONTROL_TIME_INTERVAL 0.002`（即 **500 Hz**）。
   - `AllegroHandDrv::_parseMessage()` 将 16 位整型编码器值按 `* (M_PI / 180.0) * 0.088` 转为弧度，并置位 4 指就绪掩码 `_curr_position_get`（`0x0F`）；指尖传感器数据做指数平滑（`alpha = 0.25`，限幅 `0..5000`）。
   - `AllegroHandDrv::setTorque()` 将期望力矩乘以 `1.43 * 1000` 换算后，针对 `!HAND_TYPE_A`（高减速比 V5 Plus 型号的非拇指第 2 关节索引 `1, 5, 9`）乘以 `0.5` 补偿 2 倍减速比差异；`_writeDevices()` 对每个通道指令限幅至 `[-240, 240]` 后经 `command_set_torque()` 发送 4 帧 CAN 报文。

2. **ROS 2 控制器节点（`allegro_hand_controllers`）：**
   - 基类 `AllegroNode`（`allegro_node.cpp`）：发布 `allegroHand/joint_states` 与 `allegroHand/tactile_sensors`，订阅 `allegroHand/joint_cmd`、`timechange` 与 `forcechange`；支持 `polling=true`（以 `ALLEGRO_CONTROL_TIME_INTERVAL` 对应 `500 Hz` 循环）或 `polling=false`（`create_wall_timer(1ms, ...)` 定时触发 `updateController()`）。
   - `updateController()` 主周期流程：`canDevice->readCANFrames()` → `isJointInfoReady()` 全指齐备检查 → `getJointInfo()` & 差分速度 → 虚函数 `computeDesiredTorque()` → `setTorque()` + `writeJointTorque()` → `resetJointInfoReady()` → `publishData()`；并在第 1 帧校验参数 `whichHand` / `whichType` 与硬件序列号是否匹配。
   - 派生类 `AllegroNodeGrasp`（`allegro_node_grasp.cpp`）：封装 `BHand` 控制库，订阅 `allegroHand/lib_cmd` 切换 `home` / `grasp_3` / `grasp_4` / `pinch_it` / `pinch_mt` / `envelop` / `gravcomp` / `pdControl*` / `off` 状态，通过 `pBHand->UpdateControl()` 输出 16 维关节目标力矩。
   - 包内还包含 `allegro_hand_isaacsim` 仿真联动模块，便于在同一套 ROS 2 话题结构下完成仿真验证与真机切换。

3. **V4 与 V5 仓库对比（`allegro_hand_ros_v4`）：**
   - V4 在 `allegro_node.h` 中定义 `ALLEGRO_CONTROL_TIME_INTERVAL 0.003`（**333.3 Hz**），且依赖 PEAK 专有 `/dev/pcan*` 字符设备驱动；V5 ROS 2 切换为 `0.002` s（**500 Hz**）、SocketCAN 与 Humble 原生节点模型。

---

## 对 wiki 的映射

- 实体页：[Allegro Hand](../../wiki/entities/allegro-hand.md)
- 配套硬件手册：[wonik-allegro-hand-v5-manual.md](../sites/wonik-allegro-hand-v5-manual.md)
