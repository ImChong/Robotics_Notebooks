# Δ-Flow（arXiv:2610.15794）

> 来源归档（paper）

- **标题：** Δ Flow Matching for Robot Foundation Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.15794>
- **PDF：** <https://arxiv.org/pdf/2610.15794>
- **项目页：** <https://zuo-kuangji.github.io/delta-flow/>
- **代码：** <https://github.com/NTUMARS/delta-flow-matching>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人动作为什么总在换块时卡顿？｜附代码与项目入口](../blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)（具身智能小站，2026-10-14）
- **一句话说明：** 用带轻微噪声的近期历史替代固定高斯起点，使动作与视觉生成成为残差预测，缩短生成路径并研究单步推理。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-14）：项目页链接的 GitHub 仓库可达。

## 核心摘录（公众号转述，未核验原文表格）

- 从纯噪声起步浪费了动作/视频的时间连续性；历史起点让一步推理更可行。
- 起点 = 噪声化历史；免重训替换仅限所述预训练 WAM 设置。
- 指标：在 VLA 与 WAM 上研究单步推理；数值以原文为准。

## 对 wiki 的映射

- [paper-delta-flow-matching-robot-foundation-models](../../wiki/entities/paper-delta-flow-matching-robot-foundation-models.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
