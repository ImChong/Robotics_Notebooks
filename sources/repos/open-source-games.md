# open-source-games（bobeff/open-source-games）

- **URL：** <https://github.com/bobeff/open-source-games>
- **入库日期：** 2026-09-29
- **许可：** [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/)（仓库元数据；各游戏条目链接的独立项目许可各异）
- **Stars：** ~15.5k（入库日 GitHub API）

## 一句话说明

**按品类索引的开源/源码可得商业游戏复刻清单**：README 分 Action、Racing、RTS、Sandbox 等二十余类，每条给出 **可玩入口 + 源码链接**（含引擎子链，如 Godot、Panda3D、CUBE）；**不是**单一可运行游戏或仿真器。

## 开源状态

**已开源** — 本仓为 **Markdown 策展列表**（`README.md`）；列表内各游戏/引擎为 **独立仓库**，开放程度需逐条核查（本列表只负责导航）。

## 为何值得保留

- **赛车 / 驾驶研究选型补盲：** Racing 节收录 [SuperTuxKart](https://github.com/supertuxkart/stk-code)、[TORCS](https://sourceforge.net/projects/torcs)（自述含 AI 与 research platform）、[VDrift](https://github.com/VDrift/vdrift) 等；与本库 [赛车漂移 RL 开源景观](../../wiki/overview/racing-drift-rl-open-source-landscape.md) 的 **10 仓策展** 互补——适合 **横向扫品类** 再下钻单仓。
- **引擎与完整游戏栈样本：** 多条目标注 **Godot、Panda3D、CUBE、Build** 等引擎源码链，便于对照 **可读 C++ 游戏循环 / 物理 / 网络** 与机器人侧 **自研 sim / WebGL demo**（如 [Three.js Game Skills](../../wiki/entities/threejs-game-skills.md)）的边界。
- **Living index：** 社区 PR 持续扩充；入库时以 commit 为准，**不**在 `sources/` 镜像全文。

## README 结构（归纳）

| 区块 | 与机器人 / 仿真栈的常见交叉 |
|------|------------------------------|
| Racing games | 卡丁/街机/研究向赛车 sim（STK、TORCS 等） |
| Sandbox / City-Building | 大规模交互世界、交通/建造 sim（OpenTTD 等） |
| First-Person + 引擎子节 | Godot / CUBE 等 **完整 FPS 栈** 源码 |
| Real-Time / Turn-Based strategies | 多智能体、路径与资源调度类 **可读 AI** 样本 |
| Other lists | 指向其它 awesome 类游戏索引 |

## 交叉链接

- [open-source-games 实体页](../../wiki/entities/open-source-games.md)
- [SuperTuxKart 实体页](../../wiki/entities/supertuxkart.md) — Racing 节条目
- [赛车漂移 RL 开源景观](../../wiki/overview/racing-drift-rl-open-source-landscape.md)
