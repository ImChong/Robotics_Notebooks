# nordschleife-racer

> 来源归档

- **标题：** Nordschleife Racer
- **类型：** repo
- **来源：** yassinsolim（GitHub）
- **链接：** https://github.com/yassinsolim/nordschleife-racer
- **在线演示：** https://yassin.app → [`sources/sites/yassin_app_nordschleife.md`](../sites/yassin_app_nordschleife.md)
- **Stars：** ~3（2026-08-23）
- **许可证：** MIT
- **入库日期：** 2026-08-23
- **一句话说明：** TypeScript + Three.js 浏览器 arcade-sim 竞速引擎：程序化纽北、~4900 行自研车辆物理（漂移/空飞/翻车恢复）、9 款真车参数、Supabase 多人与排行榜、幽灵回放；Playwright 物理回归。
- **代码：** https://github.com/yassinsolim/nordschleife-racer（**已开源**；车体 GLB **未入库**）
- **沉淀到 wiki：** 是 → [`wiki/entities/nordschleife-racer.md`](../../wiki/entities/nordschleife-racer.md)

---

## 核心模块（README）

| 模块 | 文件 | 职责 |
|------|------|------|
| 编排 | `Racing/RaceManager.ts` | 单人/多人状态与更新循环 |
| 物理 | `Racing/Vehicle/RaceVehicle.ts` | 悬挂、接地、转向、漂移、换挡 |
| 赛道 | `Racing/Track/NordschleifeTrack.ts` | 程序化几何与路面采样 |
| 网络 | Supabase Realtime | 多人 presence + broadcast |
| 音频 | `Racing/Audio/RaceEngineAudio.ts` | 程序化 Web Audio 引擎声 |

---

## 复现边界

- **玩：** 用 [yassin.app](https://yassin.app)（多人/榜需线上 Supabase）
- **读/改引擎：** 本仓；`Application` / `Resources` / `EventBus` 为宿主站接口，**未包含**
- **资产：** 第三方车模运行时加载，不在 git 中

---

## 关联

- 对照：[`drive_game.md`](./drive_game.md)
- 景观：[`racing_drift_rl_open_source_landscape.md`](../papers/racing_drift_rl_open_source_landscape.md)
