# God's Eye View（bilawalsidhu/gods-eye-view）

> 来源归档

- **标题：** God's Eye View（原 WorldView 系列）
- **类型：** repo（浏览器 3D 地球 + 开源空间情报层 + 语音 Agent）
- **作者：** Bilawal Sidhu 等
- **链接：** https://github.com/bilawalsidhu/gods-eye-view
- **项目页 / 入口：** https://maptheworld.ai/ — [`sources/sites/gods-eye-view-maptheworld.md`](../sites/gods-eye-view-maptheworld.md)
- **许可：** 以仓库 `LICENSE` 为准（README 强调源码可审查、可扩展）
- **入库日期：** 2026-10-01
- **Trendshift（用户触发，2026-10）：** 约 **+33.3k stars/月**；GitHub API 2026-10-01 约 **45.7k** stars
- **一句话说明：** 浏览器里的「间谍卫星模拟器」：在 photorealistic 3D 地球上叠加 **真实或公开数据源**（航班 ADS-B、船舶 AIS、卫星轨道、地震、天气、公开摄像头等），支持 click-to-track、传感器 GLSL 皮肤、Scene Director 与 **免 API key 起步** 的本地运行；内置 **实时语音 Agent** 做空间问答与白板标注。
- **为什么值得保留：** 与 [Archify](../../wiki/entities/archify.md)（描述驱动系统图）、[Agent Reach](../../wiki/entities/agent-reach.md)（外网读搜）不同，这是 **可交互、可扩展的 geospatial 前端 + 多源 live layer** 样板；对机器人/具身读者有 **空间态势、传感器 HUD、轨迹跟踪与 voice agent 编排** 的对照价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/gods-eye-view.md`](../../wiki/entities/gods-eye-view.md)

## README 要点（归纳，2026-10-01）

- **叙事：** YouTube「God's Eye View」系列病毒传播后开源；强调「看起来像禁入驾驶舱，但每一行代码可 inspect」。
- **Quick Start：** Pinokio 一键或本地终端；**可无 API key 启动**，密钥在应用内按需添加。
- **能力簇：** Cockpit 跟机、250 km 接触 roster、语音白board 多边形/路线、3D hangar 机型、FLIR/NVG 等 GLSL 皮肤、检测框 overlay、军事 HUD、Global Context、可分享 URL（含跟踪目标 handoff）、天气/GFS/雷达时间轴、卫星过境语音查询、keyless 坐标/ bundled landmark 搜索等。
- **架构：** 各数据层为独立 module，可增删；交通为真实道路 aggregate 上的模拟；部分 pose/轨迹为 coarse estimate（README 免责声明）。
- **CI：** GitHub Actions `ci.yml` badge 可见。

## 开源状态（步骤 2.5）

- **已开源** — 完整前端与 layer 模块在 GitHub；`maptheworld.ai` 与 README Quick Start 链回仓库。

## 对 wiki 的映射

| 目标 | 链接 |
|------|------|
| 实体页 | [`wiki/entities/gods-eye-view.md`](../../wiki/entities/gods-eye-view.md) |
| 架构图 Skill 对照 | [`wiki/entities/archify.md`](../../wiki/entities/archify.md) |
| Agent 外网工具 | [`wiki/entities/agent-reach.md`](../../wiki/entities/agent-reach.md) |
