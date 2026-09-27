# Khronos OpenXR 官方门户

> 来源归档

- **标题：** OpenXR — High-performance access to AR and VR platforms and devices
- **类型：** site（Khronos 标准门户 + 规范入口）
- **来源：** [The Khronos Group](https://www.khronos.org/)
- **链接：**
  - 门户：https://www.khronos.org/openxr/
  - 规范（HTML，含已注册扩展）：https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html
  - 规范 PDF：https://registry.khronos.org/OpenXR/specs/1.1/pdf/openxr.pdf
  - SDK 与教程：https://github.com/KhronosGroup/OpenXR-SDK（归档：[repos/khronos_openxr_sdk.md](../repos/khronos_openxr_sdk.md)）
  - 参考手册：https://registry.khronos.org/OpenXR/specs/1.1/man/html/
  - 一致性测试（CTS）：https://github.com/KhronosGroup/OpenXR-CTS
- **入库日期：** 2026-09-27
- **一句话说明：** **OpenXR** 是 Khronos 维护的免版税开放标准：为 AR/VR（统称 XR）提供跨厂商 **统一 C API**，覆盖 HMD、控制器、手/眼/物体/全身追踪、触觉与图形提交；OpenXR 1.1 将常用扩展并入核心以减少碎片化。
- **沉淀到 wiki：** 是 → [`wiki/entities/openxr.md`](../../wiki/entities/openxr.md)

## 为什么值得保留

- 机器人遥操作、第一人称采数、CloudXR/仿真 XR 设备栈的 **坐标系与追踪字段** 多直接或间接对齐 OpenXR（见 [XRoboToolkit](../papers/xrobotoolkit_arxiv_2508_00097.md)、[Isaac Teleop](../../wiki/entities/isaac-teleop.md)）。
- 相对各头显私有 SDK，门户 + 规范是 **一手** 定义：应用生命周期、reference space、swapchain、输入 profile、扩展机制。
- 列出 **Conformance 运行时**（Quest、PICO、SteamVR、Monado、Android XR 等），便于硬件与软件选型。

## 开源 / 公开核查（2026-09-27）

| 项 | 状态 |
|----|------|
| 规范文本 | **公开下载**（Khronos Registry；Spec 版本随 Registry 更新，门户页标注 OpenXR 1.1 主线） |
| 官方 Loader / SDK | **已开源** — Apache-2.0，见 [khronos_openxr_sdk.md](../repos/khronos_openxr_sdk.md) |
| CTS | **已开源** — GitHub `OpenXR-CTS` |
| 各厂商 Runtime | **部分开源**（如 Collabora **Monado**）；Meta/PICO/Valve 等多为预装闭源 runtime + Conformant 认证 |

## 核心摘录（门户 + 规范导读）

### 定位

- 解决 XR **碎片化**：此前需为每款设备写专有 API；OpenXR 提供 **单一高性能跨平台 API**，应用一次开发、多 runtime 移植；平台差异通过 **扩展（extension）** 与 **API Layer** 暴露。
- 标准化能力包括：HMD、控制器、基站、手/眼/物体/全身 tracker、触觉、引擎集成、云/5G 承载等。

### 典型应用生命周期（Programmer 视角，门户摘要）

1. **`xrCreateInstance`** — 连接可用 OpenXR **runtime**（Loader 负责枚举与分发）。
2. **`xrCreateSystem`** — 选择物理显示与输入/追踪/图形设备子集。
3. **创建 swapchain / 视图** — 按平台图形 API（Vulkan/D3D/OpenGL ES 等）渲染 **view**。
4. **`xrCreateSession` + 帧循环** — 获取当前/预测 **tracking pose**，读 **input**（action 建议系统），**提交帧**。

### Runtime 实现者视角

- Runtime 是在主机上实现 OpenXR API 的库，将每个 API 调用映射到底层设备驱动与合成器；Loader 在应用与多个 runtime 之间做 **分发与扩展链**。

### OpenXR 1.1（门户，2024 起主线）

- 将 **multiple vendor extensions** 合并进核心，降低「只存在于扩展表」的功能不确定性；Working Group 后续以 **定期 core 更新 + 扩展试验** 节奏演进。

### Conformant Runtimes（门户列举，节选 — 遥操作/采数常见）

| 厂商 / 项目 | 设备或范围（门户原文摘要） |
|-------------|---------------------------|
| Meta | Quest 3 / Pro / 2、Rift S、Meta XR Simulator |
| ByteDance (PICO) | Neo3、PICO 4、PICO 4 Ultra |
| Valve | SteamVR（Conformant 头显） |
| Collabora | Monado 开源 runtime |
| Google | Android XR |
| HTC | Vive Focus 3、Cosmos、Wave |
| Microsoft | HoloLens、Mixed Reality 头显 |
| Qualcomm | Snapdragon Spaces |
| NVIDIA | CloudXR 套件 leverage OpenXR（门户引述） |

完整列表以门户 **Conformant OpenXR Runtimes** 为准。

### 与 Web / 引擎

- **Unity** OpenXR Plugin（2020 LTS+）、**Unreal** 4.24+、**Godot** 4.0+ 内置、**WebXR** 默认后端等 — 机器人侧常见 **Unity Client + PC Service** 架构（如 XRoboToolkit）仍依赖底层 OpenXR runtime。

## 对 wiki 的映射

- 升格 [`wiki/entities/openxr.md`](../../wiki/entities/openxr.md) — XR 标准实体页。
- 交叉：[遥操作任务](../../wiki/tasks/teleoperation.md)、[Isaac Teleop](../../wiki/entities/isaac-teleop.md)、[XRoboToolkit 论文实体](../../wiki/entities/paper-xrobotoolkit.md)、[PICO 4 Ultra 采数](../../wiki/entities/pico-4-ultra-egocentric-capture.md)。
