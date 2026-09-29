# sc-datav（knight-L/sc-datav）

> 来源归档

- **标题：** sc-datav — Three.js 可视化大屏
- **类型：** repo（React + Vite 前端应用）
- **作者：** knight-L
- **链接：** https://github.com/knight-L/sc-datav
- **演示：** https://knight-l.github.io/sc-datav/（归档：[sites/sc-datav.md](../sites/sc-datav.md)）
- **入库日期：** 2026-09-29
- **一句话说明：** **Three.js + React 19 + ECharts** 的 3D 地图数据可视化大屏：四川省 GeoJSON 轮廓精确挤出、飞线/扫光动画、多图表联动、Leva 实时调参、autofit 多分辨率适配。
- **开源状态：** **已开源** — Apache-2.0；`pnpm dev` / `pnpm build` 可本地复现；无后端与实时数据管道（演示数据在前端静态 JSON）。
- **沉淀到 wiki：** 是 → [`wiki/entities/sc-datav.md`](../../wiki/entities/sc-datav.md)

## 为何值得保留

- **机器人/具身系统的「运营视图」层：** 机队状态、任务热力、告警趋势常与 **地理分布** 绑定；本仓给出 **d3-geo + @react-three/fiber + ECharts** 的完整拼装范例，区别于仿真视口（Isaac/MuJoCo）与 teleop HUD。
- **可复用模块边界清晰：** `pages/SCDataV/scMap.tsx`（地图）、`flyLine.tsx`（飞线）、`chart*.tsx`（侧栏图表）、`content.tsx`（布局）——便于 fork 后替换为 WebSocket/OTel 指标源。
- **配套地理资产工具链：** README 指向 [sat-hunter](https://github.com/knight-L/sat-hunter) 下载区域卫星瓦片/轮廓贴图，说明作者意图是 **可换省域/园区** 而非写死四川 demo。

## README 要点（归纳，2026-09-29）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub |
| Stars | ~2340（入库日 API） |
| 语言 | TypeScript |
| 许可 | Apache-2.0 |
| 包管理 | PNPM ≥ 8，Node ≥ 18 |
| 构建 | Vite 8 + `@vitejs/plugin-react` |

### 技术栈

| 层次 | 依赖 |
|------|------|
| UI | React 19、react-router 7、styled-components |
| 3D | three、`@react-three/fiber`、`@react-three/drei`、postprocessing |
| 2D 图表 | ECharts 6、keli-heatmap.js |
| 地理 | d3-geo、topojson-client；`src/assets/sc.json` / `sc_outline.json` |
| 动画/布局 | GSAP、autofit.js |
| 调试 | Leva |

### 目录结构（主干）

```
src/
├── assets/             # sc.json / sc_outline.json 等地理数据
├── components/         # chart、虚拟滚动等
├── pages/SCDataV/      # 大屏页：scMap、flyLine、chart1–3、content
└── App.tsx
```

### 运行命令

```bash
pnpm install
pnpm dev      # 开发
pnpm build    # 生产构建
pnpm preview  # 预览 dist
```

## 与机器人研究/工程的关联点

- **Fleet / 云边监控大屏：** 指标来自 Prometheus/OTel 或自建 API 时，可保留 Three+ECharts 壳，仅替换数据源；控制环仍应在边缘闭环（见 [边缘–云端协同](../../wiki/concepts/edge-cloud-robotics.md)）。
- **数字孪生「鸟瞰视图」：** 与 Omniverse/UE 内视不同，本仓是 **轻量 Web 大屏**；适合 NOC、展厅、管理后台，不替代仿真内相机。
- **与 Three.js Game Skills 边界：** [threejs-game-skills](../../wiki/entities/threejs-game-skills.md) 面向 **可发布游戏 + Agent QA**；本仓面向 **信息密度与地理可视化**，无 Playwright 证据链。

## 对 wiki 的映射

- 实体页：[`wiki/entities/sc-datav.md`](../../wiki/entities/sc-datav.md)
- 项目页归档：[`sources/sites/sc-datav.md`](../sites/sc-datav.md)
