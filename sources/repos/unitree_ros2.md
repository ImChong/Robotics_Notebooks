# unitree_ros2

> 来源归档（以 [unitreerobotics/unitree_ros2 `v0.3.0` Release](https://github.com/unitreerobotics/unitree_ros2/releases/tag/v0.3.0) 为准；发布日 2025-08-15）

- **标题：** unitree_ros2
- **类型：** repo
- **来源：** unitreerobotics（Unitree 官方 GitHub 组织）
- **链接：** https://github.com/unitreerobotics/unitree_ros2
- **项目页 / Release：** https://github.com/unitreerobotics/unitree_ros2/releases/tag/v0.3.0
- **对照区间：** [v0.2.0...v0.3.0](https://github.com/unitreerobotics/unitree_ros2/compare/v0.2.0...v0.3.0)
- **许可：** BSD-3-Clause
- **星标（截至 2026-08-16）：** ~798
- **最近推送：** 2026-07-02
- **主要语言：** C++
- **分类：** ROS 集成
- **入库日期：** 2026-07-24；v0.3.0 深度补全 2026-08-16
- **一句话说明：** ROS 2 直接消费 Unitree DDS 消息的官方包；v0.3.0 把 G1 双臂 / Dex3 / 高层 Arm SDK 示例补齐，并把 `unitree_hg` 手部 msg 对齐 SDK2。
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree-ros2.md`](../../wiki/entities/unitree-ros2.md)
- **组织地图：** [`sources/repos/unitree.md`](unitree.md)

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 源码 | **已开源、可运行** — `unitreerobotics/unitree_ros2`（BSD-3-Clause；`cyclonedds_ws` 消息包 + `example` 可 `colcon build`） |
| 项目页性质 | GitHub Release Notes = 本条「项目页」；无独立 `*.github.io` |
| 数据/权重 | 不适用（中间件与示例仓） |
| 边界 | README 引言仍写 Go2/B2/H1；**G1 能力以 v0.3.0 示例与 `unitree_hg` 包为准**，勿被 README 首段过时表述误导 |

---

## v0.3.0 发布要点（2025-08-15）

### Features / Examples

- **g1：** 新增 Dex3 示例，并把 DDS msg **对齐 SDK2**。
- **example：** 新增 `g1_audio_client_example`、`g1_dual_arm_example`、`g1_ankle_swing_example`、`g1_loco_client_example`、`g1_arm_action_example`、`g1_arm_sdk_dds_example`。

### Bug Fixes

- **Foxy：** `topic_datistics_collector` 不再未判断 `header` 类型就读 `headers.stamp`；v0.2.0 标为 Humble 的能力现可在 Foxy 使用。

### BREAKING CHANGE

`cyclonedds_ws/src/unitree/unitree_hg/msg/` 中下列消息已改以对齐 SDK2；**上游未给迁移脚本**，需对照字段自行改节点：

| 消息 | v0.2.0 | v0.3.0（对齐 SDK2） |
|------|--------|---------------------|
| `HandCmd.msg` | `MotorCmd[] motor_cmd` | 追加 `uint32[4] reserve` |
| `HandState.msg` | `motor_state` → `imu_state` → `press_sensor_state` → `power_v/a` → `reserve[2]` | `motor_state` → **`press_sensor_state` 提前** → `imu_state` → `power_v/a` → 新增 `system_v` / `device_v` / `error[2]` → `reserve[2]` |
| `PressSensorState.msg` | `pressure[12]` + `temperature[12]` | 追加 `lost`、`reserve` |

字段重排意味着：只按旧顺序反序列化的自定义桥会静默错位，而不是编译失败。

---

## G1 示例入口（v0.3.0 `example/src`）

| 可执行文件 | 路径 | 通信面 | 用途 |
|------------|------|--------|------|
| `g1_dual_arm_example` | `g1/lowlevel/` | `lowcmd` / `lowstate`（`unitree_hg`） | 先 `ReleaseMode`，3 s 回零，再跟踪 `.seq` 离线双臂轨迹（29-DoF；臂关节从 `LEFT_SHOULDER_PITCH=15` 起） |
| `g1_arm_sdk_dds_example` | `g1/high_level/` | 发布 `/arm_sdk`，订 `/lowstate` | 高层 Arm SDK：编译期 `G1ARM5`（13 关节）或 `G1ARM7`（17 关节）；未用关节 `q` 作权重淡入/淡出 |
| `g1_arm_action_example` | `g1/high_level/` | `/api/arm/request` · `/api/arm/response` | 预置手臂动作；API `7106` 执行 / `7107` 列表；FSM 须在 `{500, 501, 801}` |
| `g1_dex3_example` | `g1/dex3/` | `/dex3/{left\|right}/cmd` · `/lf/dex3/{left\|right}/state` | Dex3 7 电机 + 9 压感；`HandCmd` / `HandState`；交互：旋转 / 握持 / 停 / 打印 |
| `g1_loco_client_example` | `g1/high_level/` | LocoClient（FSM / 速度 / 起坐） | `--get_fsm_id`、`--set_velocity=`、`--move=`、`--wave_hand` 等 CLI |
| `g1_audio_client_example` | `g1/audio_client/` | Audio client | TTS、音量、16 kHz 单声道 PCM、LED |
| `g1_ankle_swing_example` | `g1/lowlevel/` | `unitree_hg` 低层 | 踝摆示例 |

`g1_dual_arm_example` 用法：`g1_dual_arm_example <resource_directory> [behavior_name]`，例如 `./behavior_lib/ motion`。`g1_dex3_example` 用法：`g1_dex3_example <L|R> <network_interface>`。

---

## README 要点（编译自上游，仍适用）

- 底层与 [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) 同走 CycloneDDS，ROS 2 msg 可直接通信，不必再 wrap C++ SDK。
- 已测：Ubuntu 20.04 + Foxy；Ubuntu 22.04 + **Humble（推荐）**。可用 `.devcontainer` / Dockerfile。
- `cyclonedds_ws`：消息工作空间（`unitree_go` / `unitree_api` / `unitree_hg`）。
- `example`：示例工作空间。
- Foxy 需先在**未 source ROS** 的终端编译 CycloneDDS 0.10.x；Humble 可跳过。
- 网口写入 `setup.sh` 的 `CYCLONEDDS_URI`；仿真可用 `setup_local.sh`（`lo`）。

---

## 对 wiki 的映射

- 实体页：[`wiki/entities/unitree-ros2.md`](../../wiki/entities/unitree-ros2.md) — 写入 v0.3.0 版本锚点（G1 双臂 / Dex3 / 手部 msg 对齐 SDK2）。
- 交叉：[`wiki/entities/unitree-sdk2.md`](../../wiki/entities/unitree-sdk2.md)、[`wiki/entities/unitree-g1-software-stack.md`](../../wiki/entities/unitree-g1-software-stack.md)、[`wiki/entities/unitree-dexterous-hand-services.md`](../../wiki/entities/unitree-dexterous-hand-services.md)。
- 组织枢纽：[`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)
