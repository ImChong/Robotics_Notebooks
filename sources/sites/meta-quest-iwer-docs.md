# IWER 官方文档站（Meta Quest）

> 来源归档

- **标题：** IWER Documentation — Immersive Web Emulation Runtime
- **类型：** site（项目文档 / GitHub Pages）
- **来源：** Meta Quest，`meta-quest/immersive-web-emulation-runtime`
- **链接：**
  - 文档首页：https://meta-quest.github.io/immersive-web-emulation-runtime/
  - 仓库：https://github.com/meta-quest/immersive-web-emulation-runtime
  - npm：https://www.npmjs.com/package/iwer
- **入库日期：** 2026-09-27
- **一句话说明：** IWER 官方文档：无原生 WebXR 浏览器上的 **WebXR 仿真运行时**、输入重映射、交互录制回放，以及 `@iwer/devui` / `@iwer/sem` 配套能力说明。
- **沉淀到 wiki：** 是 → [`wiki/entities/immersive-web-emulation-runtime.md`](../../wiki/entities/immersive-web-emulation-runtime.md)

## 为什么值得保留

- 机器人栈里 **WebXR 浏览器客户端**（遥操作页、CloudXR.js、仿真内嵌页）在桌面开发时依赖 **可重复的 XR API 仿真**，IWER 是 Meta Quest 组织维护的一手入口。
- 文档强调 **跨浏览器 WebXR 开发工具** 与 **action capture/playback**，与遥操作自动化测试、回归场景相关。

## 开源核查（2026-09-27）

| 项 | 状态 |
|----|------|
| 文档站 | **公开**（GitHub Pages） |
| 运行时 | **已开源** MIT — 见 [meta_quest_immersive_web_emulation_runtime.md](../repos/meta_quest_immersive_web_emulation_runtime.md) |

## 核心摘录（文档首页摘要）

- **Emulate WebXR Anywhere：** 任意浏览器解锁 WebXR 仿真，无需原生 WebXR；可定制 WebXR 开发工具链。
- **Recycle XR Controls：** 轻量高性能输入重映射层，叠在 WebXR 项目之上，提升跨平台输入复用。
- **Action Capture and Playback：** 在 XR 环境录制用户操作并在多设备回放，服务 WebXR 自动化测试。

## 对 wiki 的映射

- [`wiki/entities/immersive-web-emulation-runtime.md`](../../wiki/entities/immersive-web-emulation-runtime.md)
