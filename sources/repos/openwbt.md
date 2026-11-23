# OpenWBT

> 来源归档（国内具身开源全景 · 源码核对）

- **标题：** OpenWBT
- **类型：** repo
- **机构：** 银河通用（GALBOT · 清华大学）
- **链接：** <https://github.com/GalaxyGeneralRobotics/OpenWBT>
- **关联论文与项目页：** [R2S2 (arXiv:2505.10918)](https://www.arxiv.org/pdf/2505.10918) · <https://zzk273.github.io/R2S2/>
- **许可：** Apache-2.0
- **入库日期：** 2026-09-06
- **核查日期：** 2026-11-21
- **一句话说明：** 银河通用与清华开源的 Unitree G1 全身遥操作实现：Vision Pro（TeleVisionWrapper）与 IK 驱动上肢和灵巧手，双手柄驱动下肢 Squat/Locomotion ONNX 策略，支持真机、MuJoCo 与 Isaac Sim V4.5.0。
- **核对范围：** 通读 `GalaxyGeneralRobotics/OpenWBT@a0154ca` 的 `README.md`、`deploy/run_teleoperation_real.py`、`deploy/run_teleoperation_mujoco.py`、`deploy/controllers/controller.py`（Controller、Controller_loco、Controller_squat 初始化与主循环）及 `deploy/teleop/robot_control/robot_arm_ik.py` 接口调用；未在本机连接 G1 真机、Vision Pro 或 Isaac Sim 运行代码。
- **通读：** 已完成

## 开源状态

- **已开源（Apache-2.0）**：README TODO 列表中 R2S2 论文及 Unitree G1 的真机、MuJoCo、Isaac Sim（V4.5.0）全身遥操作已发布；Unitree H1 全身遥操作与 Latent Skill Space 仍为未勾选待办项（`[ ]`）。
- **工程定位说明：** README Clarification 明确指出 OpenWBT 主要受 R2S2 技术支撑，但**并非论文的严格复现**，而是加入了大量面向遥操作真机稳定性的工程适配技巧。

## 关键入口与实现要点（`a0154ca`）

- **真机入口：** 机器人端运行 `deploy/teleop/image_server/image_server.py`，主机端运行 `python -m deploy.run_teleoperation_real --config run_teleoperation.yaml --net eno1`（支持 `--debug` 调试模式不下发动作）。
- **仿真入口：** MuJoCo 使用 `python -m deploy.run_teleoperation_mujoco --config run_teleoperation.yaml`；Isaac Sim（V4.5.0）使用 `start_ik_server.sh` + `start_isaacsim_vision.sh`（或无 VR 版 `start_isaacsim.sh`）。
- **控制分层：** `TeleVisionWrapper` 读取头显与手腕/手部姿态，`G1_29_ArmIK.solve_ik` 解算 15–29 关节角并做每步 `±0.01 rad` 限幅；下肢由 `Runner_online_real_dexhand` 在 `SQUAT`（下蹲/俯仰/臂姿调整）与 `LOCOMOTION`（行走）两种模式间经双手柄（`/dev/ttyACM0`, `/dev/ttyACM1`）切换，分别调用 `SquatLowLevelPolicy` 与 `LocoLowLevelPolicy` ONNX 策略，经 PD 控制发送 `LowCmd`。

## 对 wiki 的映射

- [wiki/entities/cn-os-openwbt.md](../../wiki/entities/cn-os-openwbt.md)
- [wiki/entities/paper-notebook-unleashing-humanoid-reaching-potential-via-real.md](../../wiki/entities/paper-notebook-unleashing-humanoid-reaching-potential-via-real.md)（支撑论文 R2S2）
