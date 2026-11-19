# REK 官网（rek.com）

> 来源归档

- **标题：** REK — Humanoid Robot Fights
- **类型：** site（商业活动 / 赛事门户）
- **URL：** <https://rek.com/>
- **机构：** REK（Robot Embodied Kombat；旧金山）
- **创始人：** Cix Liv（VR/AR 背景；Founders Inc. 投资组合）
- **入库日期：** 2026-06-28
- **核对日期：** 2026-11-19
- **抓取方式：** 官网渲染页面与前端主包源码全文核对（`rek_home_rendered.txt` 与 `rek_main.js`，含 `/rules`、`/fights`、`/shop`、`/careers`、`/privacy`、`/terms`）
- **核对范围：** 官网首页及 SPA 全路由核对（首页 CTA、`/rules` `AUTO`/`PILOT` 分组与 `The Arbiter` 判罚规范、`/fights` 统计、`/shop` 机器人目录、`/careers` 与闭源条款 `/terms`）
- **通读：** 已完成
- **一句话说明：** REK 以 **VR 遥操作 + 全接触人形格斗** 打造现场观赛联赛并配套线上格斗模拟器：选手戴头显实时驱动改装 **Unitree G1**（及 **EngineAI T800** 等机型）在擂台互殴；官网 `/rules` 同时定义 `PILOT`（遥操）与 `AUTO`（封装自主策略）双赛道及 **500 Hz** `The Arbiter` 计分体系，且软件/服务端全面闭源。

## 官网与 SPA 路由全文核对要点（截至 2026-11-19）

| 区块 / 路由 | 内容要点（全文核对） |
|---|---|
| **首页（`/`）** | 「Real pilots, real punches, Real Steel made real」；4 个核心行动入口涵盖现场赛事/购票、REK Simulator、机器租赁与商城；公开 REK 2.1 季后赛及 11 月 20 日现场总决赛链路 |
| **赛事统计（`/fights`）** | 累计完成 **15 场真实人形格斗**（横跨 **3 个国家**）以及 **30,000+ 场线上模拟器对战**（覆盖 **65 个国家**）；已知冠军/真实驾驶员包括 **Andy**、**Osone**、**Suslike** |
| **竞赛规则（`/rules`）** | 基准机身规格：**1.32 m** 身高、**35 kg** 整备、**29 关节**、**500 Hz** 控制频率、**120 N·m** 峰值力矩；明确划分 **PILOT division**（VR 遥操机手）与 **AUTO division**（密封自主策略）；**The Arbiter** 以 500 Hz 双路信号与加速度分级判定得分，支持 `KO / TKO / STKO / DEC / DQ`，并要求 **< 100 ms** 急停与无线电截断 |
| **线上商店（`/shop`）与租赁** | 展示 **8 款在售人形机器人**及标价（含 Unitree G1 等商用机型）与活动/制片租赁服务 |
| **团队与招聘（`/careers`）** | 公开招募 **RL Researcher** 与 **VP of Events**，印证其「全身格斗控制策略 + 现场竞技赛事运营」双轨架构 |
| **开源与许可边界（`/terms` & `/privacy`）** | **完全闭源商业平台**：基于 Supabase / GCP / Stripe 与 Steam / Meta / PICO 运行；`/terms` 明确归属专有知识产权并禁止逆向工程或代码提取 |

## 第三方公开信息（交叉核对，非官网原文）

| 主题 | 要点 | 来源 |
|------|------|------|
| **全称** | Robot Embodied Kombat（REK） | [Founders Inc. 投资组合](https://f.inc/portfolio/rek/) |
| **成立** | 2025；团队约 10 人 | Founders Inc. |
| **技术栈** | 自研 **REK TEK**：人体动作 → 机器人本体映射；VR 界面含机体视角、环境、血量/性能 HUD | [Humanoid Sports Network](https://humanoidsportsnetwork.com/rek-robot-fight-tour-sells-out/) |
| **硬件** | 主力 **Unitree G1**（约 127 cm / 30 kg）；亦用 **H1-2**（约 1.8 m / 70 kg）；拳套、护具，部分演示含器械 | 同上；[Rest of World](https://restofworld.org/2026/chinese-robot-boxing-unitree-rek/) |
| **REK America** | 2025-11 五城巡演（LA / Vegas / Austin / Miami / NYC），场场售罄 | Humanoid Sports Network |
| **REK1** | 2026-02-07 **Kezar Pavilion**（旧金山）；购票者可申请成为 Pilot | [Reflex Arc 项目页](https://reflexarc.co.uk/projects/rek-robots) |
| **首场爆款** | 2025-09 旧金山 Temple 夜店 inaugural，售票近 **3400**（场馆容量约 2500）；Twitch 联创 Justin Kan、UFC 选手 Hyder Amil 等 pilot G1 | Humanoid Sports Network |
| **观赛形态** | REK0 **VR180** 可在 Apple Vision Pro（Vantage VR）沉浸式回看 | Reflex Arc |
| **产业语境** | 与 **[URKL](../../wiki/entities/urkl.md)**（EngineAI · 统一 T800 + 自主算法）等中国联赛形成对照：REK 强调 **embodied VR 选手** 而非纯自主策略 | Rest of World；本仓库 [engineai-urkl.md](./engineai-urkl.md) |

## 对 wiki 的映射

- 主实体：[REK（人形格斗联赛）](../../wiki/entities/rek.md)
- 硬件交叉：[Unitree G1](../../wiki/entities/unitree-g1.md)、[Unitree](../../wiki/entities/unitree.md)
- 任务交叉：[Teleoperation](../../wiki/tasks/teleoperation.md) — VR 全身映射的现场竞技形态
- 产业对照：[URKL](../../wiki/entities/urkl.md) — EngineAI **自主算法** 标准化 T800 联赛
- 研究对照：[RoboStriker](../../wiki/entities/paper-notebook-robostriker.md) — **自主**人形拳击多智能体 RL，与 REK **人类 pilot** 路线互补

## 待后续深读（可选）

- [ ] REK TEK 控制延迟、安全终止与力矩限幅的工程细节（若官方技术博客发布）
- [ ] G1 U2 租赁 SKU 与 SDK 文档入口（shop 开放后）
