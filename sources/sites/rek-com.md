# REK 官网（rek.com）

> 来源归档

- **标题：** REK — Humanoid Robot Fights
- **类型：** site（商业活动 / 赛事门户）
- **URL：** <https://rek.com/>
- **机构：** REK（Robot Embodied Kombat；旧金山）
- **创始人：** Cix Liv（VR/AR 背景；Founders Inc. 投资组合）
- **入库日期：** 2026-06-28
- **一句话说明：** REK 以 **VR 遥操作 + 全接触人形格斗** 打造现场观赛联赛：选手戴头显实时驱动 **Unitree G1**（及更大机型）在擂台互殴；官网同时提供 **G1 U2 租赁**（29 DoF、Full SDK）与赛事早鸟订阅。

## 官网公开要点（2026-06 抓取）

| 区块 | 内容要点 |
|------|----------|
| **品牌定位** | 「Real pilots, real punches, Real Steel made real」— 真人技能映射到实体人形格斗 |
| **REK0** | 早期里程碑赛事（旧金山），两机对打 + 现场观众验证格式 |
| **Live Events** | VR Pilots · Full Contact · Real Robots；巡演与单场购票入口 |
| **RENT A ROBOT** | **Unitree G1 U2**：活动/课堂/制片租赁；**29 DoF**、AI-Powered、**2h 电池**、**Full SDK** |
| **Shop** | 按国家登记早鸟名单，开店通知 |

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
