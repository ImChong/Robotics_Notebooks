# collabora/monado

> 来源归档

- **标题：** Monado — Open source OpenXR runtime
- **类型：** repo
- **来源：** Collabora（OpenXR 生态主要开源 runtime 维护方）
- **链接：** https://gitlab.freedesktop.org/monado/monado
- **镜像 / 社区：** 门户亦列 GitHub 生态链接；Issue 多在 freedesktop GitLab
- **Homepage：** https://monado.freedesktop.org/
- **Khronos 门户：** 列为 **Conformant** 开源 OpenXR runtime
- **入库日期：** 2026-09-27
- **一句话说明：** **Monado** 是可在 Linux（及部分 Android/Windows 实验路径）上运行的 **开源 OpenXR runtime**，适合无厂商闭源 runtime 时的开发机、CI 与驱动 bring-up；不等同于 Quest/PICO 商用 runtime，但 API 行为以 Khronos CTS 为对齐目标。
- **沉淀到 wiki：** 是 → [`wiki/entities/openxr.md`](../../wiki/entities/openxr.md)

## 开源状态（2026-09-27）

**已开源**：Monado 全栈开源（驱动 glue、state tracker、OpenXR 层）。生产遥操作仍以 **头显自带 Conformant runtime**（Meta/PICO）为主；Monado 用于 **桌面 Linux 仿真、协议调试、开源 XR 研究**。

## 为什么值得保留

- Khronos 门户 **Conformant Products** 中唯一的 prominently listed **开源 runtime**，是理解「OpenXR 实现者视角」的可读代码库。
- 与 **SteamVR OpenXR**、**Android XR** 等闭源 runtime 对照，说明 Loader + 多 runtime 共存模型。

## 对 wiki 的映射

- [`wiki/entities/openxr.md`](../../wiki/entities/openxr.md) — Runtime 选型表
- 可选深入：与 [Isaac Teleop](../../wiki/entities/isaac-teleop.md) 的 CloudXR / 本地 runtime 分工对照
