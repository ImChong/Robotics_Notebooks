# 规划轨迹为何微调不动VLA？11篇论文配套代码、模型与评测入口

> 来源归档（blog / 微信公众号）

- **标题：** 规划轨迹为何微调不动VLA？11篇论文配套代码、模型与评测入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/lN98LWBbDN7SmfN_FWcu2A
- **发表日期：** 2026-09-14
- **入库日期：** 2026-09-14
- **抓取方式：** WebFetch（Jina / wechat 工具未预装）；正文归纳归档
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md`](../raw/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)
- **一句话说明：** 11 篇具身论文盘点，覆盖统一扩散 VLA、潜接口抗捷径、TAMP 分布对齐微调、3D 部件分割、磁悬浮抓取、离线 RL 熵稳定、稀疏触觉 VTLA、铰接 in-hand 操作、衣物折叠合成数据、世界模型仿真器与空地事件感知；**11/11 独立 `paper-*` 详情节点**（本 ingest **新建 10**、**复用 1** EVPeriscope）。

## 11 篇 → 本库节点

| # | 论文 | arXiv | 开源结论 | wiki |
|---|------|-------|----------|------|
| 01 | Dynin-Robotics | [2609.13053](https://arxiv.org/abs/2609.13053) | **已开源** | [paper-dynin-robotics](../../wiki/entities/paper-dynin-robotics.md) |
| 02 | LIT | [2609.12641](https://arxiv.org/abs/2609.12641) | **已开源** | [paper-lit-latent-interface-training](../../wiki/entities/paper-lit-latent-interface-training.md) |
| 03 | DATAFARM | [2609.12316](https://arxiv.org/abs/2609.12316) | **待发布** | [paper-datafarm](../../wiki/entities/paper-datafarm.md) |
| 04 | UniPart | [2609.12898](https://arxiv.org/abs/2609.12898) | **已开源** | [paper-unipart](../../wiki/entities/paper-unipart.md) |
| 05 | Gripper MagBot | [2609.12883](https://arxiv.org/abs/2609.12883) | **部分开源** | [paper-gripper-magbot](../../wiki/entities/paper-gripper-magbot.md) |
| 06 | SCQ | [2609.12749](https://arxiv.org/abs/2609.12749) | **待发布** | [paper-scq-rl](../../wiki/entities/paper-scq-rl.md) |
| 07 | STAR | [2609.12549](https://arxiv.org/abs/2609.12549) | **待发布** | [paper-star-vtla](../../wiki/entities/paper-star-vtla.md) |
| 08 | ArtManip | [2609.12498](https://arxiv.org/abs/2609.12498) | **已开源**（[ArtGym](https://github.com/youngcv/artgym)；2026-09-15 复核） | [paper-artmanip](../../wiki/entities/paper-artmanip.md) |
| 09 | FoldNet++ | [2609.12433](https://arxiv.org/abs/2609.12433) | **待发布** | [paper-foldnet-plus-plus](../../wiki/entities/paper-foldnet-plus-plus.md) |
| 10 | Pelican-Sim 1.0 | [2609.12036](https://arxiv.org/abs/2609.12036) | **待核实** | [paper-pelican-sim](../../wiki/entities/paper-pelican-sim.md) |
| 11 | EVPeriscope | [2609.11920](https://arxiv.org/abs/2609.11920) | **已开源** | [paper-evperiscope](../../wiki/entities/paper-evperiscope.md)（复用） |

## 对 wiki 的映射

- **11/11 独立详情节点**；**0 重复 arXiv 节点**
- 阅读坐标：[VLA/TAMP 规划 11 篇技术地图](../../wiki/overview/vla-tamp-planning-11-papers-technology-map.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[Imitation Learning](../../wiki/methods/imitation-learning.md)、[Manipulation](../../wiki/tasks/manipulation.md)

## 当前提炼状态

- [x] 公众号正文抓取与归纳
- [x] 11 篇独立节点（10 新建 / 1 复用 EVPeriscope）
- [x] 项目页/仓库开源状态核查（步骤 2.5）
