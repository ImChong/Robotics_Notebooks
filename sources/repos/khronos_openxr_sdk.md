# KhronosGroup/OpenXR-SDK

> 来源归档

- **标题：** OpenXR SDK — Generated headers and sources for OpenXR loader
- **类型：** repo
- **来源：** Khronos Group（OpenXR Working Group）
- **链接：** https://github.com/KhronosGroup/OpenXR-SDK
- **Homepage / 规范：** https://www.khronos.org/openxr/（归档：[sites/khronos-openxr.md](../sites/khronos-openxr.md)）
- **Stars：** ~1.1k（2026-09）
- **许可证：** Apache-2.0（Khronos 标准实现组件常见许可）
- **入库日期：** 2026-09-27
- **一句话说明：** 官方 **OpenXR Loader** 与生成头文件/源码：应用链到系统已安装的 Conformant runtime；含示例与构建脚本，是 C/C++ 直接对接 OpenXR 的默认起点。
- **沉淀到 wiki：** 是 → [`wiki/entities/openxr.md`](../../wiki/entities/openxr.md)

## 开源状态（2026-09-27）

**已开源**：本仓提供 Loader（非完整 vendor runtime）、`include/openxr/` 头文件、示例工程。完整 HMD runtime 由 Meta/PICO/Valve/Collabora Monado 等单独提供；Loader 负责 `xrGetInstanceProcAddr` 与 runtime 动态加载。

## 仓库角色（README 摘要）

| 组件 | 作用 |
|------|------|
| **OpenXR Loader** | 应用与 ICD（Installable Client Driver / runtime）之间的标准入口 |
| **Headers** | 与 Registry 规范版本对齐的 C API |
| **HelloXR 等示例** | 最小 instance → session → 渲染循环 |
| **OpenXR-SDK-Source** | 部分生成流程在姊妹仓 `OpenXR-SDK-Source`（Registry 代码生成） |

## 与机器人 / 遥操作栈的关系

- **头显侧 Unity/Unreal** 通常不直接编译本仓，而是通过引擎 OpenXR 插件调用系统 runtime。
- **PC 侧中间层**（如 JSON 姿态流、CloudXR、Isaac Teleop Device I/O）在需要 **裸 C API** 或调试 **extension**（手追踪、composition layer）时会引用本 SDK 头文件与 Loader 行为。
- 手/控制器 pose 的 **reference space** 与 **右手系** 约定以规范为准；应用层转机器人坐标系需在中间层显式变换（见 XRoboToolkit 论文）。

## 对 wiki 的映射

- [`wiki/entities/openxr.md`](../../wiki/entities/openxr.md) — 「工程实践 / Loader vs Runtime」
- 对照 [`wiki/entities/paper-xrobotoolkit.md`](../../wiki/entities/paper-xrobotoolkit.md) — 应用层 JSON 窄腰
