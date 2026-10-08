# 公司路线：22 个未注明日期节点核查

- **类型：** 官方发布 / 论文历史 / GitHub 版本与提交证据汇编
- **核查日期：** 2026-10-08
- **范围：** 本轮 main 公司路线中 22 个空日期节点；18 个补齐事件月份，4 个保留空值。
- **路线入口：** [公司路线对照](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md)

## 日期口径

路线统一展示 YYYY-MM。优先官方 News、Release published_at 或 arXiv v1；缺少首发公告时，只将可核实的具体版本、含实现提交或官方演示作为事件，不声称它就是首次公开。GitHub 根提交可能来自迁移或私有开发，created_at / updated_at / README 当前修改日不能证明开源首发。Release 名称、正文事件日、UTC 提交时间分别注明，不把平台记录当成所有版本的共同发布日期。

机构节点只有成立公告明确到月时才采用成立事件，并调整节点标题；公司总览没有单一发布日则保留空值。

## 逐节点证据

| 公司 / 节点 | 路线日期与事件类型 | 证据与边界 |
| --- | --- | --- |
| NVIDIA / [Isaac Lab](../../wiki/entities/isaac-lab.md) | 2024-06 · 正式版本 | 2024-06-26 首个正式版本 v1.0.0；不使用继承的 Orbit 提交或 2025 年论文日期。 [官方证据](https://github.com/isaac-sim/IsaacLab/releases/tag/v1.0.0) |
| NVIDIA / [GR00T-WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md) | 2025-11 · 官方首发 | 官方 README News：2025-11-12 Initial release；当时仅 Decoupled WBC，SONIC 与 MotionBricks 后续加入。 [官方证据](https://github.com/NVlabs/GR00T-WholeBodyControl#news) |
| NVIDIA / [Cosmos Curator](../../wiki/entities/cosmos-curator.md) | 2025-07 · 正式版本 | GitHub v1.0.0 Release published_at：2025-07-25（UTC）；开源工具版本事件，不代表托管云服务首发。 [官方证据](https://github.com/NVIDIA/cosmos-curator/releases/tag/v1.0.0) |
| NVIDIA / [Cosmos Transfer](../../wiki/entities/cosmos-transfer.md) | 2025-03 · 论文 v1 | Transfer1 的 arXiv:2503.14492 v1 提交于 2025-03-18；Transfer2.5 等后续版本另计。 [官方证据](https://arxiv.org/abs/2503.14492) |
| NVIDIA / [Cosmos Cookbook](../../wiki/entities/cosmos-cookbook.md) | 2025-10 · 官方公告 | Transfer2.5 官方 README News：2025-10-28 加入 Cosmos Cookbook；不使用 12-01 后续介绍文章日期。 [官方证据](https://github.com/nvidia-cosmos/cosmos-transfer2.5#news) |
| 1X / [NEO Beta（EVE → NEO）](../../wiki/entities/1x-technologies.md) | 2024-08 · 产品发布 | 1X 官方公告：2024-08-30 发布 NEO Beta；是家庭双足原型事件，不是 EVE 首发或 NEO 商业交付日。 [官方证据](https://www.1x.tech/discover/announcement-1x-unveils-neo-beta-a-humanoid-robot-for-the-home) |
| 逐际动力 LimX / [TRON1 RL 部署](../../wiki/entities/cn-os-tron1-rl-deploy-ros2.md) | 2024-11 · 含实现提交 | 2024-11-04 官方提交含 PF/SF/WF_TRON1A 配置与 ONNX 策略；上一条 09-20 提交无 TRON，早期 06 月是 PointFoot 前序。提交时间不证明首次公开。 [官方证据](https://github.com/limxdynamics/tron1-rl-deploy-ros2/commit/5980ee3d16d56a28d6b497cec603e6151183619a) |
| RAI Institute / [RAI Institute 成立与研究路线](../../wiki/entities/rai-institute.md) | 2022-08 · 机构成立 | 成立公告正文事件日 2022-08-12，页面栏为 08-11；均为同月。当前五方向总览不是同日发布的模型。 [官方证据](https://rai-inst.com/resources/press-release/hyundai-launches-boston-dynamics-ai-institute/) |
| 星海图 Galaxea / [G0 / G0Plus 历史版本](../../wiki/entities/paper-galaxea-g05.md) | 2025-09 · 权重发布 | 固定 revision 13a16a9 的 News：G0 权重 2025-09-09、微调/推理代码 09-17；G0Plus 2026-01-04、更新 02-12。日期锚定系列 G0 起点。 [官方证据](https://github.com/OpenGalaxea/GalaxeaVLA/blob/13a16a9049aee8f1d799b56fccc0c5832a75fc2f/README.md#-news) |
| 星动纪元 ROBOTERA / [teleop_client](../../wiki/entities/cn-os-teleop-client.md) | 2025-08 · 含实现提交 | 默认分支根提交 35aaa7b：2025-08-01，含 pub_client.py 和 ROS 接口；是含实现的历史起点，不证明首次公开或产品首发。 [官方证据](https://github.com/roboterax/teleop_client/commit/35aaa7b84c27377aac3ba46f7684cd655845d608) |
| 星动纪元 ROBOTERA / [xbot_sdk_api](../../wiki/entities/cn-os-xbot-sdk-api.md) | 2026-01 · 含实现提交 | 默认分支根提交 34512a0：2026-01-17，含控制器与 Python 示例；提交时间不证明首次公开，不代表底层算法全开源。 [官方证据](https://github.com/roboterax/xbot_sdk_api/commit/34512a0338416f42024e2a7bb021b4766a545086) |
| 星动纪元 ROBOTERA / [robotera_vla：M7 基线](../../wiki/entities/cn-os-robotera-vla.md) | 2026-04 · 含实现提交 | 默认分支唯一根提交 9ff7067：2026-04-17 Initial import，含采集契约与训练/推理基线；不是 ERA-42 或 π₀.₅ 首发。 [官方证据](https://github.com/roboterax/robotera_vla/commit/9ff7067fa262f055875fe8a3e56d445734a7751c) |
| 亮源新创 Light Origins / [Lightbot 0：跑酷演示](../../wiki/entities/lightbot-0.md) | 2026-08 · 官方演示 | 2026-08-03 LightParkour 博客展示 Lightbot 0；日期对应这次官方硬件演示，不声称整机首次发布或上市。 [官方证据](https://www.lightorigins.com/blog/lightparkour) |
| 亮源新创 Light Origins / [InsightBench](../../wiki/entities/insight-bench.md) | 2026-09 · 首次公开发布 | 2026-09-09 根提交明确写 INSIGHT-Bench v1: initial public release；与 09-01 LightNav-0 博客及 08 月论文分开。 [官方证据](https://github.com/lightorigins/Light-INSIGHT-Bench/commit/4cf6f94ec357c9a4b8906e81329434975b6f3f37) |
| 萝博派对 RoboParty / [Human-Humanoid Tools / hhtools 0.1.0 Preview](../../wiki/entities/human-humanoid-tools.md) | 2026-09 · 版本发布 | GitHub Release：0.1.0(beta) / HHTools 0.1.0 Preview，2026-09-24 published_at；是 GUI 预览版本，不是首批工具链首发。 [官方证据](https://github.com/Roboparty/human-humanoid-tools/releases/tag/0.1.0(beta)) |
| 萝博派对 RoboParty / [MimicLite：PPO / ROA](../../wiki/entities/mimiclite.md) | 2026-08 · 当前版本记录 | 2026-08-31 官方提交公开列出 MimicLite-PPO / MimicLite-ROA；是当前公开版本表记录，不沿用旧 Huge/Base/v1.1 或首批工具链日期。 [官方证据](https://github.com/Roboparty/MimicLite/commit/3963976de8778d9292305fc8efacbcae79ed6685) |
| 萝博派对 RoboParty / [UFO](../../wiki/entities/roboparty-ufo.md) | 2026-07 · 框架命名与实现提交 | 2026-07-02 官方提交 Rebrand project as UFO and add FB/TLDR presets；前序 MJLab BFM-Zero 根提交为 06-30，不等于当时已发布 UFO。 [官方证据](https://github.com/Roboparty/UFO/commit/3c36e178fbb3e92adcdd76541b3cc8426fc62ea2) |
| 萝博派对 RoboParty / [TeCH](../../wiki/entities/paper-tech-humanoid-control.md) | 2026-07 · 方法实现事件 | 2026-07-13 UFO 官方提交将公开 TLDR preset 改名为 TeCH；是方法实现命名事件，不是论文 v1 或成果网页首发。 [官方证据](https://github.com/Roboparty/UFO/commit/e3679e3264e365a42259dc258fc191e67a6b7b46) |
| 银河通用 Galbot / [AstraBrain-WAM](../../wiki/entities/galbot-astrabrain.md) | 保留空日期 | 官方 about / 首页有模型说明但无单一首发日期；WBC 0.5、WAM-TTT 与 WRC 展示是不同事件，不借用其日期。 [已核查入口](https://galbot.com/about/) |
| 亮源新创 Light Origins / [Light Origins 公司与全栈](../../wiki/entities/light-origins.md) | 保留空日期 | 公司总览持续更新；已知成立于 2024 年底但官方未确认月份，不补造 YYYY-MM，也不借用融资或模型日期。 [已核查入口](https://www.lightorigins.com/en/about) |
| 蚂蚁灵波 Robbyant / [LingBot 全栈](../../wiki/entities/robbyant.md) | 保留空日期 | 公司 / 七条模型线的持续聚合入口，没有单一发布事件；各模型已有独立日期，不以公司注册或第一款模型日期替代。 [已核查入口](https://www.robbyant.com/about-robby) |
| 萝博派对 RoboParty / [Know-How / 完整研发文档](../../wiki/overview/roboparty-lab-party-os-technology-map.md) | 保留空日期 | 飞书持续维护文档未公开可核实的首发日；Party OS 发布和第三方文章日期不等于该文档或章节首发。 [已核查入口](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87) |

## 仓库历史的补充核查

- TRON1：默认分支完整历史共 29 条，最早 2024-06-05 仅 README；06 月前序是 PointFoot。09-20 树未含 TRON，下一条 11-04 树已含 PF / SF / WF_TRON1A 的配置和 policy.onnx，因此采用 11-04 含实现事件而非 06 月仓库起点。
- teleop_client / xbot_sdk_api / robotera_vla：默认分支完整历史分别 5 / 2 / 1 条，根提交无父节点，分别已有客户端与接口 / 控制器示例 / M7 基线；记录提交事件，不推定首次公开。
- UFO：前序 2026-06-30 是 MJLab BFM-Zero；07-02 命名为 UFO 并加入 FB / TLDR，07-13 公开 preset 改名 TeCH。这两个路线事件不能互相替代。
- INSIGHT-Bench：完整默认分支历史 13 条；根提交正文明确 initial public release，是本轮少数可直接确认为首次公开的提交。
- G0：固定 13a16a9 README 明确列出 2025-09-09 权重、09-17 代码、2026-01-04 G0Plus 和 02-12 后续更新。按 G0 系列起点排序，不把 G0Plus 各版本压成同一天。
