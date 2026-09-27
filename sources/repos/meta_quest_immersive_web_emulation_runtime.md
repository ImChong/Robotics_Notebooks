# meta-quest/immersive-web-emulation-runtime（IWER）

> 来源归档

- **标题：** Immersive Web Emulation Runtime（IWER）
- **类型：** repo
- **来源：** Meta Quest（GitHub 组织 `meta-quest`）
- **链接：** https://github.com/meta-quest/immersive-web-emulation-runtime
- **Homepage / 文档：** https://meta-quest.github.io/immersive-web-emulation-runtime/
- **npm：** `iwer`（最新 **2.5.0**，2026-09-24）
- **Stars：** ~107（2026-09-27）
- **许可证：** MIT
- **入库日期：** 2026-09-27
- **一句话说明：** TypeScript **WebXR Device API 仿真运行时**：在无原生 WebXR 的桌面浏览器里跑 WebXR 应用；支持输入重映射、XR 交互录制与回放，配套 `@iwer/devui` 与 `@iwer/sem`（MR 平面/网格/命中测试仿真）。
- **沉淀到 wiki：** 是 → [`wiki/entities/immersive-web-emulation-runtime.md`](../../wiki/entities/immersive-web-emulation-runtime.md)

## 开源状态（2026-09-27）

**已开源**：主包 `iwer` + 文档站 + 示例；MIT。非 Quest 原生 OpenXR runtime，而是 **浏览器侧 WebXR 仿真层**。

## README / npm 核心摘录

| 能力 | 说明 |
|------|------|
| WebXR 仿真 | 在无 WebXR 的 modern browser 中 **模拟 WebXR Device API** |
| 跨浏览器开发 | 便于 WebXR 项目 CI、桌面调试，再部署到 Quest 等真机 |
| 输入重映射 | 轻量层，可在 WebXR 项目之上做 **跨平台 XR 输入** 复用 |
| Action capture / playback | XR 环境内用户操作 **录制与回放**，支持自动化测试 |
| 配套包 | `@iwer/devui`（仿真设备 overlay UI）；`@iwer/sem`（Synthetic Environment Module：plane/mesh/hit-test 等 MR 特性仿真） |

## 安装与依赖（npm 2.5.0）

```bash
npm install iwer
```

运行时依赖：`gl-matrix`、`webxr-layers-polyfill`。

## 与机器人 / 遥操作栈的关系

- **Web 遥操作 / WebXR 客户端**：Isaac Teleop、CloudXR.js 等路径上的 **浏览器 WebXR** 原型，可在桌面用 IWER 先跑通会话与输入，再上 Quest 真机。
- **与 OpenXR 分层不同**：IWER 仿真的是 **WebXR JS API**，不是 Khronos OpenXR C runtime；Quest 真机 WebXR 仍走浏览器 + 设备 runtime。
- **与 mjswan 等**：浏览器内 RL demo 若用 WebXR 施力/手追踪，可用 IWER 做 **无头显开发机** 调试。

## 对 wiki 的映射

- 实体页 [`wiki/entities/immersive-web-emulation-runtime.md`](../../wiki/entities/immersive-web-emulation-runtime.md)
- 交叉：[`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md)、[`wiki/entities/mjswan.md`](../../wiki/entities/mjswan.md)、[`wiki/entities/isaac-teleop.md`](../../wiki/entities/isaac-teleop.md)
