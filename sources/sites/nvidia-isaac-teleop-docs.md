# Isaac Teleop 官方文档（nvidia.github.io/IsaacTeleop）

> 来源归档（ingest）

- **标题：** Welcome to Isaac Teleop
- **类型：** site（官方文档）
- **发布方：** NVIDIA
- **原始链接：** <https://nvidia.github.io/IsaacTeleop/main/index.html>
- **文档根：** <https://nvidia.github.io/IsaacTeleop/>
- **配套仓库：** <https://github.com/NVIDIA/IsaacTeleop>
- **PyPI：** `isaacteleop`
- **版本核查：** 文档站 `main` 与 `release/1.5.x`（2026-09-05）；Quick Start 建议钉 `isaacteleop[cloudxr,retargeters]~=1.5.0`
- **入库日期：** 2026-09-05
- **一句话说明：** Isaac Teleop 的架构 / 设备生态 / CloudXR Quick Start / Televiz 合成器 / 第一人称无标记手部重建文档；仿真与真机共用同一套设备接口与 retargeting 图。
- **沉淀到 wiki：** [Isaac Teleop](../../wiki/entities/isaac-teleop.md)
- **仓库归档：** [nvidia_isaac_teleop.md](../repos/nvidia_isaac_teleop.md)

## 步骤 2.5：源码开放核查（2026-09-05）

文档页头部链到 GitHub，Quick Start 走 `pip install isaacteleop`。

| 判定 | 边界 |
|------|------|
| **已开源** | [`NVIDIA/IsaacTeleop`](https://github.com/NVIDIA/IsaacTeleop) Apache-2.0；376★ / 85 forks；PyPI wheel（Linux x86_64 / aarch64，CPython 3.11–3.13）含 Televiz |
| **门禁依赖** | CloudXR Web Client SDK **首次运行 EULA**；egocentric 手部重建另需 **MANO** 学术许可 + BMC `.npy` + 第三方 Docker 权重（Dockerfile 各自许可证） |
| **未交付（文档 Upcoming）** | 纯非 XR 设备（gamepad / Gello / haply）、云仿真遥操作、向桌面或头显远程推相机流 |

## 文档要点（2026-09-05）

### 定位

统一的高保真 **egocentric + 机器人** 采数栈：标准化设备接口、图式 retargeting、MCAP 录放，并与 **ROS 2 / Isaac Sim / Isaac Lab / LeRobot** 互操作。

### 架构四块（`overview/architecture.html`）

| 块 | 作用 |
|----|------|
| Unified Device Interface | AVP / Quest / Pico 等头显；USB/BLE 手套、踏板、身体追踪；统一时间戳循环 |
| Retargeting Interface | tensor in / tensor out，可单点或整段轨迹 |
| Data Interface | FlatBuffers schema；MCAP 录放；**LeRobot 数据集互操作** |
| Visualization（Televiz） | `isaacteleop.viz`：相机/传感器平面 + 3D 重建 RGBD 合成到 XR 或桌面窗 |

### Televiz（`getting_started/televiz.html`）

- **合成器，不是采集/传输层。** 吃 GPU 帧（CuPy / PyTorch / `__cuda_array_interface__`），经 Vulkan + OpenXR + CUDA 合成。
- 四层：`QuadLayer`（相机平面）、`CylinderLayer`（宽 FOV 柱面，仅 XR）、`EquirectLayer`（360°/180°，仅 XR）、`ProjectionLayer`（gsplat / nvblox / 神经重建的 per-view `(color, depth)`）。
- 一场 session **要么** 一个 `ProjectionLayer`，**要么** 若干纹理层，二者不同时存在。
- 与 Teleop 设备追踪 **共用一个 OpenXR / CloudXR 连接**。
- 默认 XR 路径把纹理层交给 runtime 合成：无深度测试；开 `alpha_blend` 会让 CloudXR 改推整帧立体画面（带宽↑）。

### Quick Start（`getting_started/quick_start.html`）

1. `pip install 'isaacteleop[cloudxr,retargeters]~=1.0'`（跟 1.5.x 文档则钉 `~=1.5.0` + `pypi.nvidia.com` prerelease）。
2. 示例可后台拉起 CloudXR；首次下载 Web Client SDK 并接受 EULA。
3. 默认 `NV_DEVICE_PROFILE=Quest3` 覆盖 Quest / Pico / 桌面 IWER；AVP 要改 `auto-native`。
4. Web 客户端：<https://nvidia.github.io/IsaacTeleop/client>（无头显可用 IWER 仿真 Quest 3）。
5. 冒烟：`examples/teleop/python/gripper_retargeting_example_simple.py`。
6. 可选 **Brev** 云 GPU 零安装：云端跑 CloudXR + retargeting + Isaac Lab，头显只连实例 IP。

### 无标记手部重建（`references/egocentric_hand_reconstruction.html`）

- 路径：`src/postprocessing/egocentric_hand_reconstruction`（文档旧文曾写 `data-collection/`）。
- **ViPE** 估相机位姿 → **Dyn-HaMR** 重建 4D 手；单目已支持，双目在路线图。
- 参考 30 s 视频：约 100 GB RAM / 12 GB VRAM；ViPE ~7 min，Dyn-HaMR ~30 min。
- 可走本地文件或 `s3://` / `swift://`；批量用 **OSMO** `hand_reconstruction.yaml`。
- 质量直接受第一人称视频采集质量约束；OAK 插件另见设备文档。

### 设备生态（`overview/ecosystem.html`）

文档表含 AVP、Quest 2/3/3S、PICO 4 Ultra、PICO Motion Tracker、MANUS Metagloves Pro、Haptikos、OGLO Tactile Glove、Noitom、Logitech 踏板、OAK 离线采数。3Dconnexion SpaceMouse 标 Planned（issue #276）。插件进程隔离，厂商 SDK 可留在自有仓与自有许可证。

## 对 wiki 的映射

- 更新 **`wiki/entities/isaac-teleop.md`**：补 Televiz、无标记手重建、LeRobot 互操作、开源边界与 Upcoming 缺口。
- 交叉 [Isaac Lab](../../wiki/entities/isaac-lab.md)、[Isaac GR00T](../../wiki/entities/isaac-gr00t.md)、[LeRobot](../../wiki/entities/lerobot.md)、[Teleoperation](../../wiki/tasks/teleoperation.md)、[NuRec](../../wiki/entities/nvidia-nurec.md)。

## 外部参考

- [Architecture](https://nvidia.github.io/IsaacTeleop/main/overview/architecture.html)
- [Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html)
- [Televiz](https://nvidia.github.io/IsaacTeleop/main/getting_started/televiz.html)
- [Egocentric Hand Reconstruction](https://nvidia.github.io/IsaacTeleop/main/references/egocentric_hand_reconstruction.html)
- [Ecosystem](https://nvidia.github.io/IsaacTeleop/main/overview/ecosystem.html)
- [Web Client](https://nvidia.github.io/IsaacTeleop/client)
