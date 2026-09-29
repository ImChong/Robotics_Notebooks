# Tidewater（dgreenheck/tidewater）

> 来源归档（repo）

- **标题：** Tidewater
- **类型：** repo
- **作者：** dgreenheck
- **链接：** https://github.com/dgreenheck/tidewater
- **在线游玩：** https://dgreenheck.github.io/tidewater/
- **星标（截至 2026-09-29）：** ~931
- **最近推送：** 2026-09-25
- **主要语言：** JavaScript
- **许可证：** MIT
- **分类：** 浏览器 · WebGPU · 实时渲染 · 海洋/流体可视化
- **入库日期：** 2026-09-29
- **一句话说明：** 浏览器 WebGPU 热带岛钓鱼游戏：自研 WGSL 渲染引擎 + 四级联 FFT 海洋 + 浅水 swash + Hillaire 大气与体积云 + 完整钓具/经济循环；MIT 开源，GitHub Pages 即玩。
- **沉淀到 wiki：** 是 → [`wiki/entities/tidewater.md`](../../wiki/entities/tidewater.md)

---

## 开源状态（步骤 2.5，截至 2026-09-29）

| 资源 | 状态 |
|------|------|
| 源码 | **已开源** — MIT；`src/` 含 engine、ocean、sky、world、game、post、audio |
| 在线 Demo | https://dgreenheck.github.io/tidewater/ |
| 本地开发 | `npm install` → `npm run dev`（默认 http://127.0.0.1:5189） |
| 静态构建 | `npm run build` → `dist/`；`main` 推送经 `.github/workflows/deploy.yml` 部署 Pages |
| 测试 | `npm test` — headless 引擎 smoke + 游戏逻辑测试 |

**结论：确认已开源；无独立项目页，GitHub README + Pages 为复现与体验入口。**

## 技术要点（README 核对，2026-09-29）

- **栈：** 直接跑在 **WebGPU + WGSL**，自研小型渲染引擎，**无 Unity/Unreal 等框架**。
- **性能目标：** README 称 Apple M5 Pro 上约 **2560×1267 @ 60fps**；慢机 **动态降分辨率**；首访需编译数百 shader，可能 **1 分钟+**（浏览器缓存后更快）。
- **海洋：** 四 cascade **FFT 海洋**（Tessendorf 谱）、泡沫/白浪/风纹/涌浪；深度感知碎浪与 spray；**浅水 swash** 模拟；船/鲸尾迹；海底与水体 **焦散**；水线上下分割视图与出水镜头水滴。
- **天空：** Hillaire 2020 **物理大气**、日月星、体积积云/卷云与云影、God rays、遮挡感知 lens flare。
- **世界：** 岛屿地形、渔村/码头、Poly Haven 扫描资产、Microsoft Rocketbox 蒙皮 NPC、珊瑚礁鱼群、植被 impostor + dither LOD、座头鲸行为。
- **光照与后处理：** 级联阴影 + contact-hardening、SS 接触阴影、GTAO、bounce light、TAAU、运动模糊、bloom、自动曝光、夜间局部光与手电（含水下）。
- **音频：** CC0 场录 positional audio（浪、风、鸟、引擎、脚步、鲸歌、钓具等）。
- **玩法（`src/game/`）：** 纺车竿抛投/收线/张力条搏鱼、18 种加勒比鱼种（水域/深度/时段）、鱼市与 chandlery 经济、可漂移驾驶的柴油船、localStorage 进度。
- **目录：** `src/engine/` 场景图与 shader 组合；`src/ocean/`、`src/sky/`、`src/world/`、`src/post/`、`src/player/`、`src/audio/`、`tools/` 资产管线脚本。

## 对 wiki 的映射

- 实体页：[`wiki/entities/tidewater.md`](../../wiki/entities/tidewater.md)
- WebGPU 浏览器对照：[`wiki/entities/particles4all.md`](../../wiki/entities/particles4all.md)、[`wiki/entities/botlab-motioncanvas.md`](../../wiki/entities/botlab-motioncanvas.md)
- 仿真保真讨论：[`wiki/queries/simulation-physics-fidelity.md`](../../wiki/queries/simulation-physics-fidelity.md)
