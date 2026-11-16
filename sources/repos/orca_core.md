# orcahand/orca_core（官方控制栈核对笔记）

- **来源：** <https://github.com/orcahand/orca_core>
- **开源许可：** MIT License
- **核对版本：** v0.13.0+（含多硬件代数兼容与增强维护脚本）
- **核对日期：** 2026-11-14
- **核对范围：** `orcahand/orca_core` 仓库 `README.md`、`LICENSE`、`orca_core/base_hand.py`、`orca_core/hardware_hand.py`、`orca_core/hardware_hand_sensing.py`、`orca_core/hand_factory.py`、`orca_core/calibration.py`、`orca_core/maintenance/calibration_routine.py`、`orca_core/utils/cli.py`、`scripts/detect.py`、`scripts/calibrate.py`、`scripts/guided_tension.py`、`examples/main_demo.py`、`examples/demo_runner.py`
- **通读：** 已完成

---

## 开源与软件生态核查

- **开源状态：** **完全开源**（MIT License）。`orca_core` 提供了完整的底层舵机总成抽象（支持 **Dynamixel** 与 **Feetech** 两种舵机家族，v1/v2.0/v2.1.0/v2.1.1 硬件自动侦测）、全链路标定/张紧维护脚本、多模态感知接口（触觉传感器矩阵帧解析 + 关节绝对编码器闭环控制），以及完整 Mock 虚拟手与单元测试套件（`tests/`）。上层图形化助手 `orca_ui` 位于同组织独立仓库。

---

## 源码架构与核心入口核对

1. **统一抽象与工厂分发（`hand_factory.py` & `base_hand.py`）：**
   - `load_hand()` 根据连接探测结果（`detect_hand()` 询问控制板 `ORCA_INFO?` 固件能力标识 `CFG` 及总线响应）或显式 `config.yaml`，从 `_CLASS_MATRIX` 自动实例化对应手部实现类：纯开环驱动 `OrcaHand`、集成指尖触觉 `OrcaHandTouch`、关节绝对编码器闭环 `OrcaHandJointFeedback` 或全功能 `OrcaHandFull`（及各自对应的 `MockOrcaHand*` 仿真替身）。
   - `BaseHand` 封装统一的关节空间 API：`set_joint_positions()`（在 `config.clamp_joint_positions` 限幅后生成线性插值路点）、`get_joint_position()`、`pose_from_fractions()` 与 `set_neutral_position()`。
2. **生命周期与底层电机驱动（`hardware_hand.py` & `hardware/`）：**
   - 标准调用序：`hand.connect()`（建立串口并探测波特率与舵机家族）→ `hand.init_joints()`（`enable_torque()` → `set_control_mode()` 如 `current_based_position` → `set_max_current()` → 若未标定或强制要求则执行 `calibrate()` → 计算多圈绕回偏置 `_compute_wrap_offsets_dict()` → 移回 `neutral_position`）→ 实时读写与安全断开。
   - **相比早期论文版的重大工程演进：** v0.13.0 引入了面向 v2.1 硬件的交互式引导张紧脚本 `scripts/guided_tension.py`（分段保持电流使操作者拨动棘轮消除松弛）、`run_slack_test` / `run_tension_sweep` 维护例程，以及在 `calibration_routine.py` 中增加了实测行程对照 `joint_motor_travel` 基线检测——若因过紧导致提前堵转，会自动以更高重试电流重新驱动确认，并同时记录 `CalibrationResult`（含 `motor_limits_dict`、`joint_to_motor_ratios_dict`、`joint_encoder_calibration_dict`、`joint_roms_measured_dict`、`drive_current_measured_dict`）原子写回 `calibration.yaml`。
3. **典型官方运行脚本（`scripts/` & `examples/`）：**
   - `scripts/detect.py`：只读探测总线模型、硬件版本（如 `210`/`211`）、串口路径与标定完整性；
   - `scripts/guided_tension.py` / `scripts/tension.py`：先张紧肌腱（v2.1 用前者，v2.0 用后者）；
   - `scripts/calibrate.py`：关闭闭环反馈后运行开环限位堵转扫描与可选 `JointEncoderClient` 锚点采集（亦支持 `manual=True` 人工掰至机械止位）；
   - `scripts/neutral.py` & `examples/main_demo.py`：演示从配置加载、`connect_hand()`、`init_joints()` 到按 `demo_poses.yaml` 插值执行抓握与复位循环。

---

## 映射到 Wiki

- [orca-hand.md](../../wiki/entities/orca-hand.md)
