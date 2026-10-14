# ReCAP（arXiv:2610.14583）

> 来源归档（paper）

- **标题：** Remembering What Matters for Long-Horizon Robotic World Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.14583>
- **PDF：** <https://arxiv.org/pdf/2610.14583>
- **项目页：** <https://alexander-wu.github.io/recap-project-page/>
- **资源：** <https://huggingface.co/easylearning/recap-robot-models>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人动作为什么总在换块时卡顿？｜附代码与项目入口](../blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)（具身智能小站，2026-10-14）
- **一句话说明：** 冻结世界模型，只保留初始场景锚点与近期完整视觉动作块，移除陈旧生成历史，改善长时给定动作的视频预测。

## 开源状态

- **部分开源**（步骤 2.5 核查，2026-10-14）：Hugging Face 提供模型权重（easylearning/recap-robot-models）；项目页写代码即将发布，入库日未列 GitHub。

## 核心摘录（公众号转述，未核验原文表格）

- 长程滚动预测中陈旧的自生成历史会污染上下文；记忆该留什么比留多少更重要。
- 上下文 = 初始锚点 + 近期块；不重训骨干。记忆冲突解释可能与上下文长度效应混合。
- 指标：长时视频预测指标；以原文为准。

## 对 wiki 的映射

- [paper-recap-long-horizon-world-model-memory](../../wiki/entities/paper-recap-long-horizon-world-model-memory.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
