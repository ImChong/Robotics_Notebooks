# Isaac Teleop（NVIDIA）

> 来源归档

- **标题：** Isaac Teleop
- **类型：** repo + 官方文档 + Isaac Lab 集成文档
- **组织：** NVIDIA
- **代码：** https://github.com/NVIDIA/IsaacTeleop
- **官方文档：** https://nvidia.github.io/IsaacTeleop/main/index.html
- **Isaac Lab 功能页（第三方镜像，与官方 develop 文档同源结构）：** https://docs.robotsfan.com/isaaclab_official/develop/source/features/isaac_teleop.html
- **Isaac Lab 上游文档（推荐复核）：** https://isaac-sim.github.io/IsaacLab/main/source/features/isaac_teleop.html
- **入库日期：** 2026-06-02
- **一句话说明：** NVIDIA 统一的仿真与真机遥操作框架：标准化 XR/外设设备接口、图式 retargeting 管线、MCAP 录制回放，并与 Isaac Sim / Isaac Lab / ROS2 及模仿学习采数工作流衔接。
- **沉淀到 wiki：** [Isaac Teleop](../../wiki/entities/isaac-teleop.md)

---

## 核心定位（官方文档归纳）

Isaac Teleop 面向 **高保真 egocentric 与机器人数据采集**，提供：

1. **统一设备接口** — XR 头显、手套、脚踏板、身体追踪等标准化接入；插件体系可扩展 C++ 级设备驱动。
2. **灵活图式 retargeting** — `Source Nodes` → `Retargeters` → `TensorReorderer` / `OutputCombiner`，将人体/手柄跟踪映射到不同机器人 embodiment 的动作张量。
3. **仿真与真机同一栈** — 与 **ROS2**、**Isaac Sim**、**Isaac Lab** 协同；CloudXR 负责 Quest/Pico 等 WebXR 客户端，Apple Vision Pro 走原生 visionOS 客户端。
4. **数据工作流** — MCAP 录制与回放；在 Isaac Lab 侧与 `record_demos.py` 集成，示范可导出 HDF5 供 **Isaac Lab Mimic** 等 IL 管线使用。

GitHub 一句话：**The unified framework for sim & real robot teleoperation**。

---

## 与 Isaac Lab 的关系（迁移要点）

- Isaac Teleop **取代** Isaac Lab 旧版原生 XR 栈（`isaaclab.devices.openxr`）；迁移说明见 Isaac Lab 3.0 文档 **Migrating to Isaac Lab 3.0**。
- 集成入口为 **`IsaacTeleopDevice`**（`isaaclab_teleop` 包），协作体包括：
  - **XrAnchorManager** — XR anchor prim 与 `world_T_anchor` 坐标变换；
  - **TeleopSessionLifecycle** — 构建 retargeting pipeline、获取 OpenXR 句柄、每帧 `advance()` 输出 `torch.Tensor` 动作；
  - **CommandHandler** / **`poll_control_events()`** — 头显侧 start/stop/reset 控制（opaque data channel + JSON 命令）。
- **键盘 / SpaceMouse** 等仍走遗留 `isaaclab.devices`，**不属于** Isaac Teleop XR 管线。

---

## 支持设备（文档表摘要）

| 设备 | 输入模式 | 连接方式 | 备注 |
|------|----------|----------|------|
| Apple Vision Pro | 26 关节手部追踪、空间控制器 | 原生 visionOS App | 需从源码构建 Sample Client |
| Meta Quest 3 | 手柄（扳机/摇杆/握把）、手部追踪 | CloudXR.js WebXR | 浏览器客户端 |
| Pico 4 Ultra | 手柄、手部追踪 | CloudXR.js | Pico OS 15.4.4U+，HTTPS |
| Manus 手套 | 高精度手指追踪 | Isaac Teleop 插件 | 需外接腕部追踪源；自 `isaac-teleop-device-plugins` 迁移 |

---

## Retargeting 与典型控制方案

**Source Nodes：** `HandsSource`（左右手各 26 关节）、`ControllersSource`（握姿、扳机、摇杆等）。

**常用 Retargeters（`isaacteleop` 包，Isaac Lab 内置环境所用子集）：**

| Retargeter | 作用 |
|------------|------|
| `Se3AbsRetargeter` / `Se3RelRetargeter` | 跟踪 → 末端 7D 绝对位姿或 6D 增量 |
| `GripperRetargeter` | 扳机优先，否则拇指–食指捏合 → 夹爪标量 |
| `DexHandRetargeter` / `DexBiManualRetargeter` | 26 关节 → 灵巧手关节（`dex-retargeting` + URDF/YAML） |
| `TriHandMotionControllerRetargeter` | Quest 手柄 → G1 TriHand 7-DoF/手 |
| `LocomotionRootCmdRetargeter` | 摇杆 → `[vel_x, vel_y, rot_vel_z, hip_height]` |
| `TensorReorderer` | 多路输出拼成与环境 action space 一致的一维张量 |

**选型速查（Isaac Lab 文档）：**

| 任务 | 推荐输入 | 典型 Retargeters | 动作维 | 参考配置 |
|------|----------|------------------|--------|----------|
| Franka 操作 | 手柄 | `Se3Abs` + `Gripper` | 8 | `stack_ik_abs_env_cfg.py` |
| G1 全身 loco-manip | 手柄 | 双手 `Se3Abs` + `TriHand` + `LocomotionRootCmd` | 32 | `locomanipulation_g1_env_cfg.py` |
| G1 固定基座上身 | 手柄 | 双手 `Se3Abs` + `TriHand` | 28 | `fixed_base_upper_body_ik_g1_env_cfg.py` |
| GR1T2 / G1 Inspire 灵巧手 | 手部追踪 / Manus | `Se3Abs` + `DexBiManual` | 36+ | `pickplace_gr1t2_env_cfg.py` |

---

## Isaac Lab 内置 XR 遥操作环境（节选）

| Task ID | 输入 | 要点 |
|---------|------|------|
| `Isaac-Stack-Cube-Franka-IK-Abs-v0` | 右手柄 | 右握姿驱动 EE，右扳机夹爪 |
| `Isaac-PickPlace-GR1T2-Abs-v0` | 双手追踪 | 腕部 SE3 + `DexHandRetargeter` → Fourier 手 11-DoF |
| `Isaac-PickPlace-G1-InspireFTP-Abs-v0` | 双手追踪 | Inspire 手 12-DoF |
| `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` | 双手柄 | TriHand 映射 |
| `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` | 双手柄 | 上身 + 摇杆 locomotion |

演示录制示例：

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0 \
    --visualizer kit --xr
```

---

## Teleop 控制状态机

- 默认通过 UUID `uuid5(NAMESPACE_DNS, "teleop_command")` 的 OpenXR opaque channel 收发 JSON：`start teleop` / `stop teleop` / `reset teleop`。
- `DefaultTeleopStateManager` 产生 `teleop_state` 与 `reset_event`；脚本侧用 `poll_control_events(device)` 读取 `ControlEvents.is_active` / `should_reset`。
- 不需要头显控制时可设 `control_channel_uuid=None`。

---

## 性能与工程注意

- Quest 3 / Pico 4 Ultra 常见 **90 Hz** 显示；仿真 render step 宜匹配并可实时维持。
- XR 任务建议 `remove_camera_configs()` 减轻 GPU 争用。
- `pipeline_builder` 必须是 **callable**（非预构建对象），因 `@configclass` 会深拷贝可变属性。
- `RetargetingExecutionConfig(mode="pipelined")` 默认将 retargeting 放到 worker，减轻与 Isaac Lab 主循环的 GIL 争用。

---

## 对 wiki 的映射

- 新建 **`wiki/entities/isaac-teleop.md`**：NVIDIA 统一遥操作实体页（架构 Mermaid、设备/retargeting/IL 采数）。
- 更新 **`wiki/entities/isaac-lab.md`**、**`wiki/tasks/teleoperation.md`**：交叉引用，区分 XR 栈与键盘/SpaceMouse 遗留栈。

---

## 外部参考

- [NVIDIA/IsaacTeleop（GitHub）](https://github.com/NVIDIA/IsaacTeleop)
- [Isaac Teleop 官方文档](https://nvidia.github.io/IsaacTeleop/main/index.html)
- [Isaac Lab — Isaac Teleop](https://isaac-sim.github.io/IsaacLab/main/source/features/isaac_teleop.html)
