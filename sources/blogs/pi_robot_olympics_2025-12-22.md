# Moravec's Paradox and the Robot Olympics（Physical Intelligence）

> 来源归档（blog / 公司官方）

- **标题：** Moravec's Paradox and the Robot Olympics
- **类型：** blog
- **作者 / 组织：** Physical Intelligence
- **原始链接：** <https://www.pi.website/blog/olympics>
- **发表日期：** 2025-12-22
- **入库日期：** 2026-09-28
- **核对范围：** 2026-11-22 经 r.jina.ai 读取并通读官方博客 `https://www.pi.website/blog/olympics` 全文（含引言、5 大项目结果与 Moravec 悖论机制讨论）；确认文章未发布代码、权重或采集数据，也未展开对比 VLM 的具体模型名称。
- **通读：** 已完成
- **一句话说明：** 用微调后的 π₀.₆ 尝试 Benjie Holson 提出的日常操作挑战，对照没有机器人预训练的 VLM 微调。

## 开源状态（步骤 2.5，2026-09-28 / 2026-11-22 复核）

- **确认未开源**：博文是能力演示，未列代码、权重或采集数据。底模 π₀.₆ / RECAP 的训练代码见 [pistar06 归档](../papers/pistar06_arxiv_2511_14759.md)，同样未随 openpi 发布。

## 完整要点摘录

- 任务来自 Benjie Holson 的 Robot Olympics（[参考博文](https://generalrobots.substack.com/p/benjies-humanoid-olympic-games)），分 5 大项，每项设铜/银/金三级。作者强调这不是专项研究，主要工作是每项采集遥操作数据（多数任务少于 9 小时；翻袜子约 8 小时）。
- 五项结果：Event 1 全身/门（金：拉开并穿过自闭杠杆门）、Event 2 洗衣（金级套衣架因夹爪过宽物理不可行；完成银级袜子翻面与铜级叠 T 恤）、Event 3 工具使用（金：桌面捡钥匙并开锁，另跑了移动版；银：花生酱三明治；铜：喷壶+纸巾擦窗）、Event 4 指尖精细（银：套拾便袋翻面；金级剥橙因夹爪物理限制改用金属工具，作者主动判为违规不计成功）、Event 5 湿滑处理（金：水+海绵洗油锅；银：擦洗手指花生酱；铜：擦台面）。部分任务使用固定底座而非移动底座。
- 定量口径：作者明确未做 π\*₀.₆ 中的 RL 可靠性与速度优化，策略仍不稳定，自报平均成功率 **52%**、任务进度 **72%**；未做大规模机器人预训练的标准 VLM 微调基线成功率为 **0%**、平均任务进度 **9%**。
- 机制讨论：作者将 Moravec 悖论归因于“数据稀疏性”——网页文本能覆盖语义解释但无法传授无法言说的底层物理技能；通才 VLA 预训练的作用在于提供可组合的物理行为基底，使后续无需为每项新技能重采海量数据。
- **对 wiki 的映射：** [pi-robot-olympics](../../wiki/entities/pi-robot-olympics.md)
