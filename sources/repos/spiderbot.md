# GitHub 仓库：ERC-BPGC/SpiderBot

- **仓库：** https://github.com/ERC-BPGC/SpiderBot
- **关联项目：** Spiderbot（12-DOF 四连杆被动重力补偿开源六足机器人）
- **开源状态：** 已开源（包含 SolidWorks 装配体与 STEP 硬件图纸、`mjlab` 闭链 MJCF 与训练配置、`sim2sim` 独立 MuJoCo 验证脚本以及 `sim2real` 实机部署脚本与 `scservo_sdk`）
- **核对范围：** 已通读官方仓库 `ERC-BPGC/SpiderBot` 下的 `README.md`、`README_mjlab.md`、`pyproject.toml`、`src/mjlab/asset_zoo/robots/spiderbot/spiderbot_constants.py`、`src/mjlab/tasks/velocity/config/spiderbot/env_cfgs.py`、`sim2sim/test.py` 与 `sim2real/hardware_deploy.py`
- **通读：** 已完成

## 代码结构与核心模块

1. **闭链资产与 `mjlab` 训练配置（`spiderbot_constants.py` / `env_cfgs.py`）**
   - `HEXAPOD_XML` 加载 `Hexapod_V2_full_position.xml`（包含四连杆 `equality` 约束与 `tendon` 弹簧元件）。
   - 执行器配置包含延迟缓冲（`MIN_LAG = 1`, `MAX_LAG = 15`，`BuiltinPositionActuatorCfg`），驱动 12 个主动关节（6 条腿的 `calf_motor_link_joint_leg_i` 与 `parallel_link_top_joint_leg_i`）。
   - `env_cfgs.py` 注册平地与崎岖地形速度跟踪任务 `Mjlab-Velocity-Flat-Spiderbot` 与 `Mjlab-Velocity-Rough-Spiderbot`（同时提供 `Spider3d` 3-DOF 对照模型），对 Actor 移除 `base_lin_vel` 与 `height_scan`，对关节位置/速度施加 `delay_min_lag=1..6` 与 `history_length=3` 历史帧堆叠。

2. **Sim-to-Sim 独立 MuJoCo 校验（`sim2sim/test.py`）**
   - 以 **40 Hz**（`0.005s` 物理步长 × `decimation=5`）运行 `onnxruntime` 推理，`ACTION_SCALE = 0.25`，使用 `ObservationHistory(3)` 构建含 `imu_ang_vel`、`projected_gravity`、关节偏差、关节速度（`× 0.05`）与上一动作历史及单帧速度指令的多帧观测，并经 `PROCESSED_ACTION_CLIP` 按膝/平行连杆差异化限幅输出位置目标。
   - **注意点**：仓库默认路径指向 `sim2sim/policies/spiderbot.onnx`，实际部署前需替换为用户训练导出的策略权重。

3. **Sim-to-Real 双串口实机部署（`sim2real/hardware_deploy.py`）**
   - 依赖内置 `scservo_sdk` 同时打开双 USB 串口：`/dev/ttyACM0`（`scscl`，6 个 SC15 髋舵机 `ID_SC = [6, 3, 1, 5, 2, 4]`）与 `/dev/ttyACM1`（`sms_sts`，6 个 ST3215 大腿舵机 `ID_ST = [7..12]`），波特率 `1_000_000` bps，主控循环 **50 Hz**，预设 `ACTION_SCALE = 0.5`。
   - **代码实情与论文差异**：`hardware_deploy.py` 默认实现为 **无 IMU** 兼容推理模式（自动依据 ONNX 输入维度选择 `27` 维 `[joint_pos(12), last_action(12), cmd(3)]` 或 `39` 维追加有限差分 `joint_vel(12)`），注释亦标明 `"the original t8.onnx is not bundled"` 且可在读总线延时过高时回退用 `last_action` 替代实时回读；同时保留了 `sim2real/cpg/` 传统三脚/波浪/波动步态测试脚本及 `Mechanical Design/Hexapod Assembly/` 机械工程文件。
