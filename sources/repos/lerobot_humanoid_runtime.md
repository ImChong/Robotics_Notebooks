# lerobot-humanoid-runtime

> 来源归档

- **标题：** lerobot-humanoid-runtime（仓内包名 `lerobot_humanoid_runtime`）
- **类型：** repo
- **链接：** https://github.com/huggingface/lerobot-humanoid-runtime
- **许可证：** Apache-2.0
- **入库日期：** 2026-09-28
- **一句话说明：** 12-DoF **无双臂**双足人形的 MuJoCo 仿真、真机 CAN MIT 控制（含安全机制）、策略推理与 **LeRobot Robot 集成**；含校准与诊断工具链。
- **代码：** https://github.com/huggingface/lerobot-humanoid-runtime（**已开源**）
- **硬件 BOM：** [lerobot-humanoid-hardware](lerobot_humanoid_hardware.md)
- **沉淀到 wiki：** [lerobot-humanoid](../../wiki/entities/lerobot-humanoid.md)

---

## 平台与依赖（README）

- Raspberry Pi 5、Ubuntu、Python 3.13
- 环境管理：**uv**（`uv sync` + `--extra sim|policy|imu|viz|gamepad|tools|full`）

## 主要模块

| 路径 | 职责 |
|------|------|
| `robot/sim_robot.py` | MuJoCo 仿真控制 |
| `robot/bipedal_robot.py` | 真机 CAN MIT + 安全 |
| `control/rl_agent.py` | ONNX/Torch 策略推理 |
| `imu/IMU_integration.py` | BNO055/BNO085/JY901/mock |
| `lerobot_humanoid_lerobot_integration/` | LeRobot 硬件适配 |
| `tools/scan_motors.py` | CAN 电机扫描 |
| `tools/interactive_zeroing.py` | 交互式关节归零 |
| `tools/imu_calibration_tool.py` | IMU 零偏校准 |
| `tools/joint_nudge_tester.py` | 单关节方向验证 |
| `tools/data_acquisition.py` | 辨识用数据集采集 |

## 安全（README 强调）

- 先 `state_only` 校验状态再使能控制；保留硬件急停；勿关闭控制器安全检查。

## 对 wiki 的映射

- [LeRobot Humanoid](../../wiki/entities/lerobot-humanoid.md)、[LeRobot](../../wiki/entities/lerobot.md)
