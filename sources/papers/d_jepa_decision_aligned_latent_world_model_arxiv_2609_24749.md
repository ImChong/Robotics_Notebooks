# D-JEPA（arXiv:2609.24749）

> 来源归档（paper）

- **标题：** D-JEPA: A Decision-Aligned Latent World Model
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24749>
- **PDF：** <https://arxiv.org/pdf/2609.24749>
- **项目页：** <https://nebulis-lab.com/D-JEPA/>
- **代码：** <https://github.com/NEBULIS-Lab/D-JEPA>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 用实际执行结果学习候选未来之间的关系，修正潜在距离与任务成败不一致的问题，让世界模型选对能成功的动作。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：仓库可达（git ls-remote），项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 预测接近目标的潜在状态不等于选对动作；潜在距离与成败可能不一致。
- 以真实执行结果监督候选未来之间的排序关系，使潜在空间与决策对齐。
- 指标：公众号提示：决策对齐在匹配候选集合下评估，效果须连同动作候选生成方式理解。

## 对 wiki 的映射

- [paper-d-jepa-decision-aligned-latent-world-model](../../wiki/entities/paper-d-jepa-decision-aligned-latent-world-model.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
