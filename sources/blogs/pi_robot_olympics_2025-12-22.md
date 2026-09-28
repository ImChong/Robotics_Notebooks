# Moravec's Paradox and the Robot Olympics（Physical Intelligence）

> 来源归档（blog / 公司官方）

- **标题：** Moravec's Paradox and the Robot Olympics
- **类型：** blog
- **作者 / 组织：** Physical Intelligence
- **原始链接：** <https://www.pi.website/blog/olympics>
- **发表日期：** 2025-12-22
- **入库日期：** 2026-09-28
- **一句话说明：** 用微调后的 π₀.₆ 尝试 Benjie Holson 提出的日常操作挑战，对照没有机器人预训练的 VLM 微调。

## 开源状态（步骤 2.5，2026-09-28）

- **确认未开源**：博文是能力演示，未列代码、权重或采集数据。底模 π₀.₆ / RECAP 的训练代码见 [pistar06 归档](../papers/pistar06_arxiv_2511_14759.md)，同样未随 openpi 发布。

## 核心摘录（归纳，非全文）

- 任务来自 Holson 的 Robot Olympics（涂花生酱、洗锅、插钥匙、袜子翻面等），分五项，每项有铜/银/金。作者称这不是专项研究，主要工作是每项采集数据，多数任务少于 9 小时（翻袜子约 8 小时）。
- 五项里三项达到金级、两项银级。两项金级任务受夹爪几何限制做不到；剥橙用了金属工具，作者明确不计成功。部分任务用固定底座，原文设定是移动机器人。
- 作者自报平均成功率 52%、任务进度 72%，且没有为冲成功率去做文中提到的 RL 可靠性优化。不用 π₀.₆、只微调标准 VLM 的对照：没有任务成功，平均进度 9%。
- **对 wiki 的映射：** [pi-robot-olympics](../../wiki/entities/pi-robot-olympics.md)
