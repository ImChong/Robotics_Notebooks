# 代码真的能帮部署吗？FluxVLA、JEPLO与DIDO等12篇机器人论文资源整理

> 来源归档（blog / 微信公众号）

- **标题：** 代码真的能帮部署吗？FluxVLA、JEPLO与DIDO等12篇机器人论文资源整理
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/nsAslK7HCyhUaViGkSVgWA
- **发表日期：** 2026-09-16
- **入库日期：** 2026-09-16
- **抓取方式：** wechat-article-for-ai（Camoufox）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md`](../raw/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md)
- **一句话说明：** 12 篇具身论文盘点，覆盖 VLA 工程部署、LiDAR 足式、一步 WAM 蒸馏、对象级 DiT、系统韧性、人形安全过滤与语义通信等；**12/12 独立详情节点**（本 ingest **新建 10**、**复用 2**）。

## 12 篇 → 本库节点

| # | 论文 | arXiv | 开源结论 | wiki |
|---|------|-------|----------|------|
| 01 | FluxVLA Engine | [2609.17210](https://arxiv.org/abs/2609.17210) | **已开源** | [fluxvla-engine](../../wiki/entities/fluxvla-engine.md)（**复用**） |
| 02 | JEPLO | [2609.15770](https://arxiv.org/abs/2609.15770) | **已开源** | [paper-jeplo](../../wiki/entities/paper-jeplo.md) |
| 03 | DIDO | [2609.15570](https://arxiv.org/abs/2609.15570) | **已开源** | [paper-dido-wam](../../wiki/entities/paper-dido-wam.md) |
| 04 | SlotDiT | [2609.17414](https://arxiv.org/abs/2609.17414) | **待发布** | [paper-slotdit](../../wiki/entities/paper-slotdit.md) |
| 05 | RobResilience | [2609.17349](https://arxiv.org/abs/2609.17349) | **已开源** | [paper-robresilience](../../wiki/entities/paper-robresilience.md) |
| 06 | Machine Zygote | [2609.17300](https://arxiv.org/abs/2609.17300) | **已开源** | [paper-machine-zygote](../../wiki/entities/paper-machine-zygote.md) |
| 07 | WholeBodyWAM | [2609.16644](https://arxiv.org/abs/2609.16644) | **待发布** | [paper-wholebodywam](../../wiki/entities/paper-wholebodywam.md) |
| 08 | ProxiDex | [2609.16586](https://arxiv.org/abs/2609.16586) | **待发布** | [paper-proxidex](../../wiki/entities/paper-proxidex.md) |
| 09 | ResSafe | [2609.15988](https://arxiv.org/abs/2609.15988) | **待发布** | [paper-ressafe](../../wiki/entities/paper-ressafe.md)（**复用**） |
| 09 | Goal-Oriented Comms for Physical AI | [2609.15895](https://arxiv.org/abs/2609.15895) | **待发布** | [paper-goal-oriented-comms-physical-ai](../../wiki/entities/paper-goal-oriented-comms-physical-ai.md) |
| 10 | WLA³ | [2609.15870](https://arxiv.org/abs/2609.15870) | **待发布** | [paper-wla3](../../wiki/entities/paper-wla3.md) |
| 11 | StereoPatch | [2609.15509](https://arxiv.org/abs/2609.15509) | **部分开源** | [paper-stereopatch](../../wiki/entities/paper-stereopatch.md) |

## 对 wiki 的映射

- **12/12 独立详情节点**；**0 重复 arXiv 节点**
- 阅读坐标：[VLA 部署 12 篇技术地图](../../wiki/overview/vla-deploy-12-papers-technology-map.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Safety Filter](../../wiki/concepts/safety-filter.md)、[Locomotion](../../wiki/tasks/locomotion.md)

## 当前提炼状态

- [x] 公众号正文抓取
- [x] 12 篇独立节点（10 新建 / 2 复用）
- [x] 项目页/仓库开源状态核查（步骤 2.5）
