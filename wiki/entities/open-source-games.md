---
type: entity
tags: [open-source, game-development, curated-list, racing, simulation, game-engine]
status: complete
updated: 2026-09-29
related:
  - ../overview/racing-drift-rl-open-source-landscape.md
  - ./supertuxkart.md
  - ./threejs-game-skills.md
  - ./drive-game.md
  - ./carla.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/open-source-games.md
summary: "bobeff/open-source-games：CC0 开源游戏 mega-list（~15k stars）；按品类链到可玩站点与源码，含 STK/TORCS 等赛车与 Godot/Panda3D 引擎链；供机器人侧横向发现完整游戏栈，非 RL 训练后端。"
---

# open-source-games（bobeff）

**[bobeff/open-source-games](https://github.com/bobeff/open-source-games)** 是 GitHub 上维护的 **开源与商业游戏源码复刻索引**（CC0 列表仓，入库日 ~15.5k stars）。正文为单文件 `README.md`，按 **Action / Racing / RTS / Sandbox / …** 分节；每条通常含 **游玩链接 + `[[source]]` 源码**，部分条目再链 **游戏引擎** 独立仓库。

## 一句话定义

> **品类化的「可玩 + 可读源码」游戏电话本**：帮助从 **完整开源游戏** 反查引擎与实现，**不**提供统一 API、训练 Gym 或机器人仿真接口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| STK | SuperTuxKart | 列表 Racing 节卡丁车游戏；见 [SuperTuxKart](./supertuxkart.md) |
| TORCS | The Open Racing Car Simulator | 列表 Racing 节；自述含 AI 与 research platform 用途 |
| RTS | Real-Time Strategy | 列表 Real-Time strategies 品类 |
| CC0 | Creative Commons Zero | 本 **索引仓** 的许可；各游戏项目许可独立 |
| FPS | First Person Shooter | 列表 First-Person 品类；常附引擎源码子链 |
| RL | Reinforcement Learning | 机器人训练范式；本列表 **不** 等价于 RL 环境注册表 |

## 为什么重要

- **与本站「赛车 / 漂移 RL 景观」分工：** [赛车漂移 RL 开源景观](../overview/racing-drift-rl-open-source-landscape.md) 策展 **10 个可直接 fork 的研究向仓库**（f1tenth_gym、CARLA、drift_drl 等）；本列表在 **Racing games** 等节提供 **更广的品类扫面**（含 STK、TORCS、VDrift、Yorg/Panda3D 等），适合 **「还有没有可读赛车栈？」** 的第一站。
- **完整游戏 vs 仿真器：** 列表条目多为 **GPL/独立许可的成品游戏** 或 **引擎复刻**；相对 [CARLA](./carla.md) 的 **AD 传感器与场景 API**，或 [drive-game](./drive-game.md) 的 **浏览器自研物理**，选型时需假设 **自行封装、无标准 Gym**。
- **引擎源码链：** City-Building / FPS 等节多次出现 **Godot、Panda3D、CUBE、Build** 引擎链接——对 **图形、输入、网络、内容管线** 工程有参照价值，与 [Three.js Game Skills](./threejs-game-skills.md) 的 **Web 代理交付** 路线正交但同属 **可交互 3D 软件** 生态。
- **维护模式：** 社区 PR 扩表；本库只归档 **来源与机器人相关读法**，不镜像全文。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | GitHub Markdown 策展列表（非 runnable monorepo） |
| **许可** | 列表仓 **CC0-1.0**；各游戏见各自仓库 |
| **开源** | **已开源** — 索引与链接公开；条目指向的代码仓需逐条核查 |
| **更新** | 以 GitHub `README.md` commit 为准 |

## 与本库已有实体的映射（Racing 节摘录）

| 列表条目 | 本站下钻 | 读法提示 |
|----------|----------|----------|
| SuperTuxKart | [SuperTuxKart](./supertuxkart.md) | 趣味卡丁物理；非 RL Gym |
| TORCS | （暂无独立实体页） | 自述 **AI / research platform**；老派 C++ 赛车 sim |
| VDrift、Yorg 等 | — | 可作 **物理与引擎** 源码阅读；训练需自建接口 |

更多 **可训 / 可复现论文栈** 见 [赛车漂移 RL 开源景观](../overview/racing-drift-rl-open-source-landscape.md)。

### 选型流程（流程总览）

```mermaid
flowchart TD
  A[目标：找开源赛车/游戏栈] --> B{要 RL 训练 API?}
  B -->|是| C[本站赛车漂移景观<br/>f1tenth / CARLA / drift_drl 等]
  B -->|否，要完整游戏或引擎源码| D[open-source-games README<br/>按品类浏览]
  D --> E{条目类型}
  E -->|已有 wiki 实体| F[跳转 STK 等实体页]
  E -->|仅列表链接| G[打开 [[source]] 仓<br/>读许可与构建说明]
  C --> H[Sim2Real / 传感器 / 奖励设计]
  G --> I[自封装或当引擎样本]
```

## 局限与风险

- **无统一质量门槛：** 列表不保证构建通过、文档完整或活跃维护；fork 前看 issue/最近 commit。
- **许可混杂：** CC0 只覆盖 **索引文字**；游戏资产与引擎 **GPL / 专有数据** 等需单独合规。
- **非机器人专用：** 不应与 [CARLA](./carla.md)、Isaac 等 **机器人仿真平台** 混为一谈；[Sim2Real](../concepts/sim2real.md) 迁移成本需自评。

## 与其他页面的关系

- [赛车漂移 RL 开源景观](../overview/racing-drift-rl-open-source-landscape.md) — 研究向 10 仓地图
- [SuperTuxKart](./supertuxkart.md) — Racing 节代表条目（双仓 GPL 游戏）
- [Three.js Game Skills](./threejs-game-skills.md) — 浏览器游戏代理交付；非本列表范围
- [Sim2Real](../concepts/sim2real.md) — 从游戏/sim 迁真机时的域差概念

## 参考来源

- [open-source-games 仓库归档](../../sources/repos/open-source-games.md)

## 推荐继续阅读

- [bobeff/open-source-games README（Racing games 锚点）](https://github.com/bobeff/open-source-games#racing-games)
- [TORCS 项目页](https://torcs.sourceforge.net/) — 列表中的 research-oriented 赛车 sim
- [Godot Engine](https://github.com/godotengine/godot) — 列表多条 FPS/城建条目引用的引擎源码
