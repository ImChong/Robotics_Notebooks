# 视觉聚焦让策略更省数据：10篇机器人论文，附代码与项目入口

> 来源归档（blog / 微信公众号）

- **标题：** 视觉聚焦让策略更省数据：10篇机器人论文，附代码与项目入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/5sh8p1tClXU75U9MB5rDsQ
- **发表日期：** 2026-09-09
- **入库日期：** 2026-09-09
- **抓取方式：** WebFetch（Jina / wechat 工具链不可用）；正文归纳入库
- **一句话说明：** 汇总 10 篇近期具身/机器人论文，主线从视觉局部读出、模型式 RL 价值目标、智能体记忆压缩到关系操作片段、4D 交互预测与 3D waypoint；**10/10 均有独立 `paper-*` 详情节点**（本 ingest **新建 10**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：这批工作把「视觉/交互/记忆/空间中间表示」从隐式黑盒改成可训练接口——FocusPool 用状态条件查询读中间 CNN 局部性；CAST 让规划器经验进入 state-value 目标；MemForest 用 EventTree 压缩长程记忆；FOCI / 3DWay / Coherent4D 分别在关系片段、3D waypoint、连续 4D where-to-how 上给出结构化中间量。

### 10 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | FocusPool | [2609.08408](https://arxiv.org/abs/2609.08408) | **已开源** GitHub | [paper-focuspool](../../wiki/entities/paper-focuspool.md) |
| 02 | CAST | [2609.08853](https://arxiv.org/abs/2609.08853) | **未开源** 仅项目页 | [paper-cast-mbrl](../../wiki/entities/paper-cast-mbrl.md) |
| 03 | MemForest | [2609.08273](https://arxiv.org/abs/2609.08273) | **已开源** GitHub | [paper-memforest](../../wiki/entities/paper-memforest.md) |
| 04 | Motion-based messaging | [2609.08920](https://arxiv.org/abs/2609.08920) | **部分** 项目页列 Code 链 | [paper-motion-based-messaging](../../wiki/entities/paper-motion-based-messaging.md) |
| 05 | FOCI Policy | [2609.08743](https://arxiv.org/abs/2609.08743) | **未开源** 仅项目页 | [paper-foci-policy](../../wiki/entities/paper-foci-policy.md) |
| 06 | HiBRIDGE | [2609.08678](https://arxiv.org/abs/2609.08678) | **待核实** 公众号 GitHub 链 404 | [paper-hibridge-dialogue](../../wiki/entities/paper-hibridge-dialogue.md) |
| 07 | From Where to How | [2609.08636](https://arxiv.org/abs/2609.08636) | **待核实** 项目页 | [paper-from-where-to-how](../../wiki/entities/paper-from-where-to-how.md) |
| 08 | AURORA | [2609.08493](https://arxiv.org/abs/2609.08493) | **未开源** 匿名项目页 | [paper-aurora-hand-reconstruction](../../wiki/entities/paper-aurora-hand-reconstruction.md) |
| 09 | ReMoMask-2 | [2609.08365](https://arxiv.org/abs/2609.08365) | **已开源** GitHub | [paper-remomask-2](../../wiki/entities/paper-remomask-2.md) |
| 10 | 3DWay | [2609.08224](https://arxiv.org/abs/2609.08224) | **已开源** GitHub ECCV 2026 | [paper-3dway](../../wiki/entities/paper-3dway.md) |

## 对 wiki 的映射

- **10/10 独立详情节点**；阅读坐标见 [视觉聚焦与数据效率 10 篇技术地图](../../wiki/overview/visual-focus-data-efficiency-10-papers-technology-map.md)。
- 交叉：[模仿学习](../../wiki/methods/imitation-learning.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)、[VLA](../../wiki/methods/vla.md)、[Manipulation](../../wiki/tasks/manipulation.md)。

## 当前提炼状态

- [x] 公众号正文抓取与归纳
- [x] 10 篇独立节点（0 重复 arXiv）
- [x] 项目页/仓库开源状态核查（步骤 2.5）
