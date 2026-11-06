# 只在虚拟房间练习，人形机器人走进陌生房间能做什么？

> 来源归档（blog / 微信公众号）

- **标题：** 只在虚拟房间练习，人形机器人走进陌生房间能做什么？
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/bS_fZx7sMtiE2B4Yd8NNVA
- **发表日期：** 2026-11-06
- **入库日期：** 2026-11-06
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md`](../raw/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)
- **一句话说明：** 本期主线是把规划、仿真训练和评测的耗时与失败边界拆开看：cuRoboV2-Fast 把规划耗时与真机抓放周期分开计时，HALO 把虚拟房间里的训练带到真机，Open-HumanWAM 检验人类第一人称视频对陌生场景的帮助；其余覆盖交互式世界模型、导航与闭环评测。深读 cuRoboV2-Fast，跟进 HALO 与 Open-HumanWAM。 共 20 篇；**20/20 独立详情节点**（本 ingest **新建 16**、**复用 4**）。

## 20 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 运动与控制 | cuRoboV2-Fast | [2611.06913](https://arxiv.org/abs/2611.06913) | **已开源** | [paper-curobov2-fast](../../wiki/entities/paper-curobov2-fast.md) |
| 02 | 运动与控制 | HALO | [2611.06908](https://arxiv.org/abs/2611.06908) | **已开源** | [paper-halo-perceptive-humanoid-motion-sim](../../wiki/entities/paper-halo-perceptive-humanoid-motion-sim.md)（**复用**） |
| 03 | 操作策略与数据 | Open-HumanWAM | [2611.06817](https://arxiv.org/abs/2611.06817) | **部分开源** | [paper-open-humanwam](../../wiki/entities/paper-open-humanwam.md) |
| 04 | 交互式世界模型 | Cosmos Interactive World Simulation | [2611.06627](https://arxiv.org/abs/2611.06627) | **已开源** | [paper-cosmos-interactive-world-simulation](../../wiki/entities/paper-cosmos-interactive-world-simulation.md)（**复用**） |
| 05 | 导航与闭环评测 | Robocasa-Mem | [2611.06475](https://arxiv.org/abs/2611.06475) | **已开源** | [paper-robocasa-mem](../../wiki/entities/paper-robocasa-mem.md) |
| 06 | 导航与闭环评测 | Failure-Boundary Profiles | [2611.06453](https://arxiv.org/abs/2611.06453) | **已开源** | [paper-failure-boundary-profiles](../../wiki/entities/paper-failure-boundary-profiles.md) |
| 07 | 操作策略与数据 | PyRUA-REAL | [2611.06768](https://arxiv.org/abs/2611.06768) | **已开源** | [paper-pyrua-real](../../wiki/entities/paper-pyrua-real.md) |
| 08 | 操作策略与数据 | Latent Noise Representation Bridging | [2611.07275](https://arxiv.org/abs/2611.07275) | **已开源** | [paper-latent-noise-representation-bridging](../../wiki/entities/paper-latent-noise-representation-bridging.md) |
| 09 | 运动与控制 | Learned Safe Sets | [2611.06196](https://arxiv.org/abs/2611.06196) | **已开源** | [paper-learned-safe-sets](../../wiki/entities/paper-learned-safe-sets.md) |
| 10 | 交互式世界模型 | Spatial-UMI-Dream | [2611.07229](https://arxiv.org/abs/2611.07229) | **部分开源** | [paper-spatial-umi-dream](../../wiki/entities/paper-spatial-umi-dream.md) |
| 11 | 操作策略与数据 | Tactile Cues Controlled Study | [2611.06673](https://arxiv.org/abs/2611.06673) | **部分开源** | [paper-tactile-cues-controlled-study](../../wiki/entities/paper-tactile-cues-controlled-study.md) |
| 12 | 操作策略与数据 | RL Generalist Policy Improvement | [2611.06905](https://arxiv.org/abs/2611.06905) | **已开源** | [paper-rl-generalist-policy-improvement](../../wiki/entities/paper-rl-generalist-policy-improvement.md) |
| 13 | 操作策略与数据 | Astra on RoboMME | [2611.06184](https://arxiv.org/abs/2611.06184) | **已开源** | [paper-astra-on-robomme](../../wiki/entities/paper-astra-on-robomme.md) |
| 14 | 导航与闭环评测 | STARE | [2611.06089](https://arxiv.org/abs/2611.06089) | **已开源** | [paper-stare](../../wiki/entities/paper-stare.md) |
| 15 | 导航与闭环评测 | Commodity-Price Closed-Loop Evaluation | [2611.07441](https://arxiv.org/abs/2611.07441) | **已开源** | [paper-commodity-price-closed-loop-evaluation](../../wiki/entities/paper-commodity-price-closed-loop-evaluation.md) |
| 16 | 导航与闭环评测 | MessyNav | [2611.07417](https://arxiv.org/abs/2611.07417) | **待发布** | [paper-messynav](../../wiki/entities/paper-messynav.md) |
| 17 | 导航与闭环评测 | VABAR | [2611.06881](https://arxiv.org/abs/2611.06881) | **已开源** | [paper-vabar-variational-automation-benchmark](../../wiki/entities/paper-vabar-variational-automation-benchmark.md)（**复用**） |
| 18 | 运动与控制 | Soccerbot | [2611.06564](https://arxiv.org/abs/2611.06564) | **已开源** | [paper-soccerbot-latent-motion-priors](../../wiki/entities/paper-soccerbot-latent-motion-priors.md)（**复用**） |
| 19 | 运动与控制 | Fluidic Body | [2611.06068](https://arxiv.org/abs/2611.06068) | **已开源** | [paper-fluidic-body](../../wiki/entities/paper-fluidic-body.md) |
| 20 | 交互式世界模型 | ReactWorld | [2611.06120](https://arxiv.org/abs/2611.06120) | **项目页未列 GitHub** | [paper-reactworld](../../wiki/entities/paper-reactworld.md) |

## 对 wiki 的映射

- **20/20 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-11-06）
- [x] 20 篇独立节点（16 新建 / 4 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
