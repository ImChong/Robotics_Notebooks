# drive-game

> 来源归档

- **标题：** Nürburgring Drive（drive-game）
- **类型：** repo
- **来源：** esc5221 / bwchoi（GitHub）
- **链接：** https://github.com/esc5221/drive-game
- **在线演示：** https://drive-game.pages.dev → [`sources/sites/drive_game_pages_dev.md`](../sites/drive_game_pages_dev.md)
- **Stars：** ~11（2026-08-23）
- **许可证：** MIT
- **入库日期：** 2026-08-23
- **一句话说明：** 第一人称纽北驾驶模拟器：Three.js 渲染 + 手写 240 Hz 车辆物理（射线悬挂、Pacejka 联合滑移、离合弹射、气动与天气 grip）；OSM 真实赛道几何 + DEM 高程；Web 与 Android（Capacitor）同仓。
- **代码：** https://github.com/esc5221/drive-game（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/drive-game.md`](../../wiki/entities/drive-game.md)

---

## 核心定位

- **赛道：** 纽北 20.7 km（OSM + DEM）、Spa、练习场
- **物理：** 240 Hz；Pacejka 轮胎；路面/天气 grip；AudioWorklet 引擎声
- **功能：** 幽灵圈、动态走线、昼夜/雨雾、多机位
- **文档：** [game_logic.html](https://drive-game.pages.dev/data/game_logic.html)、[making 构建日志](https://drive-game.pages.dev/making)

---

## 典型入口

```bash
npm install
npm run dev      # http://localhost:8741
npm run build    # → dist/
```

---

## 关联

- 对照：[`nordschleife_racer.md`](./nordschleife_racer.md) — 另一纽北浏览器引擎
- 景观：[`racing_drift_rl_open_source_landscape.md`](../papers/racing_drift_rl_open_source_landscape.md)
