# sc-datav（knight-l.github.io/sc-datav）

> 来源归档

- **标题：** sc-datav — 数据可视化大屏在线演示
- **类型：** site（GitHub Pages 演示站）
- **作者：** knight-L
- **链接：** https://knight-l.github.io/sc-datav/
- **代码：** https://github.com/knight-L/sc-datav（归档：[repos/sc-datav.md](../repos/sc-datav.md)）
- **入库日期：** 2026-09-29
- **一句话说明：** 基于 Three.js + React 19 + ECharts 的 **3D 地理轮廓大屏** 在线 demo（多路由 `#/demo0`–`#/demo3`），展示飞线、扫光与图表联动。
- **沉淀到 wiki：** 是 → [`wiki/entities/sc-datav.md`](../../wiki/entities/sc-datav.md)

## 开源核查（2026-09-29）

| 项 | 状态 |
|----|------|
| 演示站 | **公开可读**（GitHub Pages） |
| 主仓库 | **已开源** — Apache-2.0；https://github.com/knight-L/sc-datav |
| 配套工具 | **已开源** — 地图轮廓/卫星瓦片下载 [sat-hunter](https://github.com/knight-L/sat-hunter)（README 链出，未单独 ingest） |

## 演示路由（门户）

| 路由 | 说明 |
|------|------|
| `#/demo0` | https://knight-l.github.io/sc-datav/#/demo0 |
| `#/demo1` | https://knight-l.github.io/sc-datav/#/demo1 |
| `#/demo2` | https://knight-l.github.io/sc-datav/#/demo2 |
| `#/demo3` | https://knight-l.github.io/sc-datav/#/demo3 |

## 为什么值得保留

- **队级/园区级监控 UI 参考：** 机器人 fleet、工厂数字孪生、物流枢纽常需要 **地理底图 + 3D 挤出 + 时序图表** 同屏；本演示把 **GeoJSON 轮廓、Three.js 场景、ECharts 面板** 捆在同一 React 应用，可作为 [可观测性](../../wiki/concepts/observability-logs-metrics-tracing.md) 与 [边缘–云端协同](../../wiki/concepts/edge-cloud-robotics.md) 的 **前端样板**（非 OTel 后端）。
- **与 Agent Skills 栈互补：** 若用 [Three.js Game Skills](../../wiki/entities/threejs-game-skills.md) 做交互游戏，本仓展示 **大屏信息架构、autofit 缩放、Leva 调参** 等「运营视图」模式。
- **高 star 社区验证：** 主仓 ~2.3k stars（入库日 API），说明 **政务/园区风 3D 地图大屏** 需求在 Web 前端侧有稳定受众。

## 对 wiki 的映射

- 实体页：[`wiki/entities/sc-datav.md`](../../wiki/entities/sc-datav.md)
- 交叉：[`wiki/entities/threejs-game-skills.md`](../../wiki/entities/threejs-game-skills.md)、[`wiki/concepts/observability-logs-metrics-tracing.md`](../../wiki/concepts/observability-logs-metrics-tracing.md)
