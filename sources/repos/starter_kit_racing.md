# starter-kit-racing

> 来源归档

- **标题：** Starter Kit Racing（mrdoob JS/Three.js 移植）
- **类型：** repo
- **来源：** mrdoob（Three.js 作者）/ Kenney 资产
- **链接：** https://github.com/mrdoob/starter-kit-racing
- **在线演示：** https://mrdoob.github.io/Starter-Kit-Racing/ → [`sources/sites/starter_kit_racing_pages_dev.md`](../sites/starter_kit_racing_pages_dev.md)
- **Stars：** ~298（2026-09-01）
- **许可证：** MIT
- **入库日期：** 2026-09-01
- **一句话说明：** Kenney「Starter Kit Racing」Godot 4.6 项目的 **纯 JavaScript + Three.js** 移植：GridMap 模块化赛道、`crashcat` 刚体碰撞、程序化引擎/撞击音效与 `editor.html` 赛道编辑器；零构建链，CDN 直引 three + crashcat。
- **代码：** https://github.com/mrdoob/starter-kit-racing（**已开源** MIT）
- **沉淀到 wiki：** 是 → [`wiki/entities/starter-kit-racing.md`](../../wiki/entities/starter-kit-racing.md)

---

## 核心定位

- **上游：** [KenneyNL/Starter-Kit-Racing](https://github.com/KenneyNL/Starter-Kit-Racing)（Godot 4.6）
- **物理：** `crashcat` 刚体；车体为 **sphere body**；墙体/弯道用 cuboid 近似 Godot `ConcavePolygonShape3D`
- **渲染：** Three.js **0.185.1**（importmap CDN）
- **赛道：** `Track.js` GridMap 铺砖（cell 9.99 × scale 0.75）；`editor.html` 可导出自定义 `map` URL 参数
- **音频：** `EngineWorklet` 程序化引擎；`ImpactSound` 撞击 one-shot；`skid.ogg` 样本 + 音调变化
- **资产：** Kenney CC0 GLB / sprites

---

## 典型入口

```bash
# 无 npm 依赖：任意静态服务器或直接打开 index.html
cd starter-kit-racing
python3 -m http.server 8080
# → http://localhost:8080/index.html
# 自定义赛道：index.html?map=<editor 导出>
# 编辑器：editor.html
```

---

## 关联

- 对照：[`drive_game.md`](./drive_game.md) — 纽北 Pacejka 高保真模拟器
- 对照：[`nordschleife_racer.md`](./nordschleife_racer.md) — TS 纽北漂移引擎
- 景观：[`racing_drift_rl_open_source_landscape.md`](../papers/racing_drift_rl_open_source_landscape.md)
