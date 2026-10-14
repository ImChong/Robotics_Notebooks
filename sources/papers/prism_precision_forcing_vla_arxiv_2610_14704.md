# PRISM (Precision Forcing)（arXiv:2610.14704）

> 来源归档（paper）

- **标题：** PRISM: Precision Forcing for Generalist Vision-Language-Action Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.14704>
- **PDF：** <https://arxiv.org/pdf/2610.14704>
- **项目页：** <https://seen-e.github.io/PRISM/>
- **代码：** <https://github.com/seen-e/PRISM>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人动作为什么总在换块时卡顿？｜附代码与项目入口](../blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)（具身智能小站，2026-10-14）
- **一句话说明：** 按精度需求对流匹配监督重加权，并把精度需求写入条件表示，让训练更关注容差敏感时刻。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-14）：项目页链接的 GitHub 仓库可达。与库内同名 PRISM 节点为不同论文。

## 核心摘录（公众号转述，未核验原文表格）

- 流匹配损失平均分摊到所有时刻，连接器插拔、拧螺丝等关键瞬间得不到足够监督。
- 精度权重 + 条件编码；实验涉及连接器与拧螺丝。
- 指标：以原文为准。

## 对 wiki 的映射

- [paper-prism-precision-forcing-vla](../../wiki/entities/paper-prism-precision-forcing-vla.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
