# 9篇开源论文看懂具身智能新动向

> 来源归档（blog / 微信公众号）

- **标题：** 9篇开源论文看懂具身智能新动向
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/CXOf3PU8-H6OzI77vnhZMA
- **发表日期：** 2026-08-23
- **入库日期：** 2026-08-23
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_9_papers_open_source_2026-08-23.md`](../raw/wechat_embodied_station_9_papers_open_source_2026-08-23.md)
- **一句话说明：** 汇总 9 篇近期具身/机器人论文（文内均给项目页或代码链），主线从「更长上下文 VLA」到「更可控动作分布」，再到「真实硬件反馈」；**9/9 均有独立 `paper-*` 详情节点**（本 ingest **新建 8**、**复用 1 既有 complete**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：这批论文共同追问机器人策略如何从离线数据、长上下文与真实交互中获得更强可执行性与鲁棒性。**SparkVLA**、open-loop action chunking 与 **StructRL** 重新审视动作 chunk、闭环响应与探索噪声；**GigaBrain-0.7** 把具身基础模型扩展到三系统架构与 3.7 万小时异构数据；**GAINS**、**ReForce**、**Neural GCS** 与 **YOPO-MINCO** 分别覆盖人类介入、力觉重定向、规划加速与安全约束。

### 9 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | SparkVLA | [2608.16172](https://arxiv.org/abs/2608.16172) | **已开源** GitHub 仓 | [paper-sparkvla](../../wiki/entities/paper-sparkvla.md) |
| 02 | Revisiting Open-Loop Execution | [2608.15938](https://arxiv.org/abs/2608.15938) | **未开源**（复用既有节点） | [paper-revisiting-open-loop-action-chunking](../../wiki/entities/paper-revisiting-open-loop-action-chunking.md) |
| 03 | GigaBrain-0.7 | [2608.15875](https://arxiv.org/abs/2608.15875) | **部分开源**：`giga-brain-0` 仓已链；预训练权重 **Coming soon** | [paper-gigabrain-0-7](../../wiki/entities/paper-gigabrain-0-7.md) |
| 04 | Dual-Head Coordination | [2608.15748](https://arxiv.org/abs/2608.15748) | **已开源** GitHub 仓 | [paper-dual-head-coordination](../../wiki/entities/paper-dual-head-coordination.md) |
| 05 | YOPO-MINCO | [2608.15741](https://arxiv.org/abs/2608.15741) | **已开源** YOPO 仓 `YOPO-MINCO` 分支 | [paper-yopo-minco](../../wiki/entities/paper-yopo-minco.md) |
| 06 | GAINS | [2608.15707](https://arxiv.org/abs/2608.15707) | **已开源** `nuomizai/HIL-RL` | [paper-gains](../../wiki/entities/paper-gains.md) |
| 07 | ReForce | [2608.15560](https://arxiv.org/abs/2608.15560) | **未开源**：项目页仅 Paper/arXiv | [paper-reforce](../../wiki/entities/paper-reforce.md) |
| 08 | Neural GCS | [2608.15440](https://arxiv.org/abs/2608.15440) | **已开源** `RIVeR-Lab/neural-graphs-of-convex-sets` | [paper-neural-gcs](../../wiki/entities/paper-neural-gcs.md) |
| 09 | StructRL | [2608.15139](https://arxiv.org/abs/2608.15139) | **待发布**：项目页无 GitHub 链 | [paper-structrl](../../wiki/entities/paper-structrl.md) |

### 文内要点速记

1. **SparkVLA** — Stop 与 action-prefix 统一排序；Anchor-Conditioned Context Encoding；RoboCerebra 47.12%。
2. **Revisiting Open-Loop** — 长 open-loop 主因短上下文模仿非马尔可夫专家；足够 context 后 reactive 最优。
3. **GigaBrain-0.7** — 三系统架构 + 37k 小时异构数据；System-3 世界模型进决策回路。
4. **Dual-Head Coordination** — 双 flow-matching 头协调机制与 runtime collapse certificate。
5. **YOPO-MINCO** — 两段 MINCO、多同伦预测、barrier 代价与 ranking loss 改造 YOPO。
6. **GAINS** — 分布 RL 建模不一致人类干预；比 RLIF 任务成功率高 22%。
7. **ReForce** — 力觉重定向：运动学 residual + 仿真力跟踪器；纸杯/夹钳接触任务。
8. **Neural GCS** — GAT 替代昂贵凸松弛；100% 成功率下最高两个数量级加速。
9. **StructRL** — 动作空间结构化探索；缓解 flow-VLA 在线 RL 的 Structured Noise Dilution。

## 对 wiki 的映射

- **9/9 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 8** 个实体；**Revisiting Open-Loop** 在先前 ingest 已有 complete 页 → **只回链博客，不重复造页**。
- 阅读坐标：[VLA 可执行性与鲁棒性 9 篇技术地图](../../wiki/overview/vla-robustness-9-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[VLA](../../wiki/methods/vla.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[模仿学习](../../wiki/methods/imitation-learning.md)、[Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 9 篇独立节点核查（8 新建 / 1 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
