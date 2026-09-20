# 机器人控制遇到约束冲突，怎样保证动作还能继续？｜附代码与评测入口

> 来源归档（blog / 微信公众号）

- **标题：** 机器人控制遇到约束冲突，怎样保证动作还能继续？｜附代码与评测入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/RozDRLth62xgulo4ccIBMw
- **发表日期：** 2026-09-20
- **入库日期：** 2026-09-20
- **抓取方式：** WebFetch（Camoufox 工具链不可用时的兜底）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md`](../raw/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md)
- **一句话说明：** 11 篇具身论文盘点，覆盖约束 QP、世界模型导航、VLM in-context、力感知操作、人形 WAM、异常检测基准、3D 动力学、液体运输、任务导向灵巧抓取与多人形协作；**11/11 独立详情节点**（本 ingest **新建 6**、**复用 5**）。

## 11 篇 → 本库节点

| # | 论文 | arXiv | 开源结论 | wiki |
|---|------|-------|----------|------|
| 01 | ElastiQP | [2609.19080](https://arxiv.org/abs/2609.19080) | **已开源** | [paper-elastiqp](../../wiki/entities/paper-elastiqp.md)（**新建**） |
| 02 | WAVE-Go | [2609.18193](https://arxiv.org/abs/2609.18193) | **已开源** | [paper-wave-go](../../wiki/entities/paper-wave-go.md)（**复用**） |
| 03 | GPT-Policy | [2609.19138](https://arxiv.org/abs/2609.19138) | **已开源** | [paper-gpt-policy](../../wiki/entities/paper-gpt-policy.md)（**复用**） |
| 04 | Dreaming the Sound of Contact | [2609.19137](https://arxiv.org/abs/2609.19137) | **待发布** | [paper-dreaming-sound-of-contact](../../wiki/entities/paper-dreaming-sound-of-contact.md)（**新建**） |
| 05 | WholeBodyWAM | [2609.18197](https://arxiv.org/abs/2609.18197) | **待发布** | [paper-wholebodywam-unimotion-4k](../../wiki/entities/paper-wholebodywam-unimotion-4k.md)（**复用**） |
| 06 | FIERCE | [2609.18651](https://arxiv.org/abs/2609.18651) | **已开源** | [paper-fierce](../../wiki/entities/paper-fierce.md)（**复用**） |
| 07 | RoboVAD | [2609.17843](https://arxiv.org/abs/2609.17843) | **部分开源** | [paper-robovad](../../wiki/entities/paper-robovad.md)（**新建**） |
| 08 | PointZero | [2609.19142](https://arxiv.org/abs/2609.19142) | **已开源** | [paper-pointzero](../../wiki/entities/paper-pointzero.md)（**新建**） |
| 09 | Fetch My Beer | [2609.18119](https://arxiv.org/abs/2609.18119) | **待发布** | [paper-fetch-my-beer](../../wiki/entities/paper-fetch-my-beer.md)（**新建**） |
| 10 | OpenDexGrasp | [2609.18117](https://arxiv.org/abs/2609.18117) | **待发布** | [paper-opendexgrasp](../../wiki/entities/paper-opendexgrasp.md)（**新建**） |
| 11 | decMHT | [2609.17824](https://arxiv.org/abs/2609.17824) | **待发布** | [paper-decentralized-multi-humanoid-pickup](../../wiki/entities/paper-decentralized-multi-humanoid-pickup.md)（**复用**） |

## 对 wiki 的映射

- **11/11 独立详情节点**；**0 重复 arXiv 节点**
- 阅读坐标：[约束控制 11 篇技术地图](../../wiki/overview/constraint-control-11-papers-technology-map.md)
- 交叉：[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

## 当前提炼状态

- [x] 公众号正文抓取
- [x] 11 篇独立节点（6 新建 / 5 复用）
- [x] 项目页/仓库开源状态核查（步骤 2.5）
