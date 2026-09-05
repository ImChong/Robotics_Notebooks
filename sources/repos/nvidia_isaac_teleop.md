# Isaac Teleop（NVIDIA）

> 来源归档

- **标题：** Isaac Teleop
- **类型：** repo + 官方文档 + Isaac Lab 集成文档
- **组织：** NVIDIA
- **代码：** https://github.com/NVIDIA/IsaacTeleop
- **官方文档：** https://nvidia.github.io/IsaacTeleop/main/index.html
- **文档归档：** [nvidia-isaac-teleop-docs.md](../sites/nvidia-isaac-teleop-docs.md)
- **Isaac Lab 功能页（第三方镜像，与官方 develop 文档同源结构）：** https://docs.robotsfan.com/isaaclab_official/develop/source/features/isaac_teleop.html
- **Isaac Lab 上游文档（推荐复核）：** https://isaac-sim.github.io/IsaacLab/main/source/features/isaac_teleop.html
- **许可：** Apache-2.0
- **规模（2026-09-05 GitHub API）：** 376★ / 85 forks；`pushed_at` 2026-09-05；Python 为主
- **首次入库：** 2026-06-02
- **复核日期：** 2026-09-05
- **一句话说明：** NVIDIA 统一的仿真与真机遥操作框架：标准化 XR/外设设备接口、图式 retargeting 管线、MCAP 录制回放，并与 Isaac Sim / Isaac Lab / ROS2 / LeRobot 及模仿学习采数工作流衔接；1.5.x 文档补齐 Televiz 合成器与第一人称无标记手重建。
- **沉淀到 wiki：** [Isaac Teleop](../../wiki/entities/isaac-teleop.md)

---

## 步骤 2.5：源码开放核查（2026-09-05）

文档站与 GitHub README 均指向本仓；Quick Start 用 PyPI `isaacteleop`，不必从源码构建即可跑 gripper 冒烟。

| 判定 | 边界 |
|------|------|
| **已开源** | Apache-2.0 仓 + PyPI wheel（Linux x86_64 / aarch64，CPython 3.11–3.13）；示例在 `examples/teleop/python/` |
| **门禁 / 第三方条款** | CloudXR Web Client SDK **首次运行 EULA**；手部重建需 **MANO** 学术账号 + BMC + 容器内第三方权重（各自许可证） |
| **未交付** | README Upcoming：非 XR 主设备、云仿真遥操作、向桌面/头显远程推相机流 |

---

## 核心定位（官方文档 + README 归纳）

Isaac Teleop 面向 **高保真 egocentric 与机器人数据采集**，提供：

1. **统一设备接口** — XR 头显、手套、脚踏板、身体追踪等标准化接入；插件体系可扩展 C++ 级设备驱动（厂商 SDK 可留在自有仓）。
2. **灵活图式 retargeting** — `Source Nodes` → `Retargeters` → `TensorReorderer` / `OutputCombiner`，将人体/手柄跟踪映射到不同机器人 embodiment 的动作张量。
3. **仿真与真机同一栈** — 与 **ROS2**、**Isaac Sim**、**Isaac Lab**（3.0+）协同；CloudXR 负责 Quest/Pico 等 WebXR 客户端，Apple Vision Pro 走原生 visionOS 客户端（Sample Client ≥ v3.0.0）。
4. **数据工作流** — FlatBuffers schema；MCAP 录制与回放；**LeRobot 数据集互操作**；Isaac Lab 侧与 `record_demos.py` 集成，示范可导出 HDF5 供 **Isaac Lab Mimic** 等 IL 管线使用。
5. **Televiz（`isaacteleop.viz`）** — 把机器人相机/传感器与 gsplat / nvblox 等 RGBD 合成到 XR 头显；与追踪共用一条 CloudXR 连接。
6. **无标记手重建** — 第一人称视频 → ViPE + Dyn-HaMR → 4D 手与相机位姿（单目已支持，双目在路线图）。

GitHub 一句话：**The unified framework for sim & real robot teleoperation**。

---

## README 用例（2026-09-05）

**已支持：**

- XR 头显做夹爪 / 三指手操作
- XR + 手套做灵巧手
- 坐姿全身 loco-manipulation（Homie）
- 基于跟踪的全身 loco-manipulation + GR00T-WholeBodyControl（SONIC）
- 第一人称采数（aka “no-robot”）

**Upcoming：** 纯非 XR 设备（gamepad / Gello / haply 等）、云仿真遥操作、向桌面或头显远程推相机流。

---

## 与 Isaac Lab 的关系（迁移要点）

- Isaac Teleop **取代** Isaac Lab 旧版原生 XR 栈（`isaaclab.devices.openxr`）；迁移说明见 Isaac Lab 3.0 文档 **Migrating to Isaac Lab 3.0**。
- 集成入口为 **`IsaacTeleopDevice`**（`isaaclab_teleop` 包），协作体包括：
  - **XrAnchorManager** — XR anchor prim 与 `world_T_anchor` 坐标变换；
  - **TeleopSessionLifecycle** — 构建 retargeting pipeline、获取 OpenXR 句柄、每帧 `advance()` 输出 `torch.Tensor` 动作；
  - **CommandHandler** / **`poll_control_events()`** — 头显侧 start/stop/reset 控制（opaque data channel + JSON 命令）。
- **键盘 / SpaceMouse** 等仍走遗留 `isaaclab.devices`，**不属于** Isaac Teleop XR 管线（文档标 3Dconnexion SpaceMouse 为 Planned，issue #276）。

---

## 支持设备（文档表摘要，2026-09-05 复核）

| 设备 | 输入模式 | 连接方式 | 备注 |
|------|----------|----------|------|
| Apple Vision Pro | 26 关节手部追踪、空间控制器 | 原生 visionOS App | 需从源码构建 Sample Client ≥ v3.0.0；CloudXR profile `auto-native` |
| Meta Quest 2 / 3 / 3S | 手柄（扳机/摇杆/握把）、手部追踪 | CloudXR.js WebXR | 浏览器客户端；默认 `Quest3` profile |
| Pico 4 Ultra | 手柄、手部追踪 | CloudXR.js | Pico OS 15.4.4U+，HTTPS |
| PICO Motion Tracker | 全身追踪 | Web Client | 需 Enterprise 或开通企业功能 |
| Manus / Haptikos / OGLO | 手指 / 触觉 | Teleop 插件 | OGLO 默认关，编译时打开 |
| Logitech Rudder Pedals | 三轴脚踏 | USB 通用踏板插件 | |
| OAK | 离线第一人称视频 | USB 3 插件 | 供手重建管线 |

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

独立冒烟（不必先装 Lab）：

```bash
pip install 'isaacteleop[cloudxr,retargeters]~=1.5.0'
python examples/teleop/python/gripper_retargeting_example_simple.py
```

---

## Teleop 控制状态机

- 默认通过 UUID `uuid5(NAMESPACE_DNS, "teleop_command")` 的 OpenXR opaque channel 收发 JSON：`start teleop` / `stop teleop` / `reset teleop`。
- `DefaultTeleopStateManager` 产生 `teleop_state` 与 `reset_event`；脚本侧用 `poll_control_events(device)` 读取 `ControlEvents.is_active` / `should_reset`。
- 不需要头显控制时可设 `control_channel_uuid=None`。

---

## 性能与工程注意

- Quest 3 / Pico 4 Ultra 常见 **90 Hz** 显示；仿真 render step 宜匹配并可实时维持。Web 客户端默认 Quest **72 FPS / 25 Mbps**、Pico 4 Ultra **90 FPS / 100 Mbps**；拉高码率易在 5 GHz Wi-Fi 上数分钟后 reprojection 抖动。
- XR 任务建议 `remove_camera_configs()` 减轻 GPU 争用。
- `pipeline_builder` 必须是 **callable**（非预构建对象），因 `@configclass` 会深拷贝可变属性。
- `RetargetingExecutionConfig(mode="pipelined")` 默认将 retargeting 放到 worker，减轻与 Isaac Lab 主循环的 GIL 争用。
- Televiz `submit()` 是 latest-wins mailbox、同步拷贝；每层单生产者。XR 默认 runtime 合成层无深度；`alpha_blend` 会关掉 CloudXR 的分层编码省带宽路径。
- 手重建参考 30 s：Ubuntu 24.04、约 100 GB RAM / 12 GB VRAM；质量受第一人称视频约束。

---

## 对 wiki 的映射

- 更新 **`wiki/entities/isaac-teleop.md`**：补 Televiz、无标记手重建、LeRobot、开源边界。
- 交叉 **`wiki/entities/isaac-lab.md`**、**`wiki/entities/isaac-gr00t.md`**、**`wiki/entities/lerobot.md`**、**`wiki/tasks/teleoperation.md`**、**`wiki/entities/nvidia-nurec.md`**、**`wiki/entities/gr00t-wholebodycontrol.md`**。

---

## 外部参考

- [NVIDIA/IsaacTeleop（GitHub）](https://github.com/NVIDIA/IsaacTeleop)
- [Isaac Teleop 官方文档](https://nvidia.github.io/IsaacTeleop/main/index.html)
- [Isaac Lab — Isaac Teleop](https://isaac-sim.github.io/IsaacLab/main/source/features/isaac_teleop.html)
- [Televiz](https://nvidia.github.io/IsaacTeleop/main/getting_started/televiz.html)
- [Egocentric Hand Reconstruction](https://nvidia.github.io/IsaacTeleop/main/references/egocentric_hand_reconstruction.html)
