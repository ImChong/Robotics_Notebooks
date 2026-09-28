# The Physical Intelligence Layer（Physical Intelligence）

> 来源归档（blog / 公司官方）

- **标题：** The Physical Intelligence Layer
- **类型：** blog
- **作者 / 组织：** Physical Intelligence；伙伴段落分别由 Weave、Ultra 撰写
- **原始链接：** <https://www.pi.website/blog/partner>
- **发表日期：** 2026-02-24
- **入库日期：** 2026-09-28
- **一句话说明：** 把 π₀ 到 π\*₀.₆ 描述成可复用的机器人基础模型层，并转述 Weave 洗衣与 Ultra 打包的现场部署。

## 开源状态（步骤 2.5，2026-09-28）

- **确认未开源**：本文是合作叙事，没有新的代码或权重入口。文中 π₀.₆ / 伙伴数据进预训练的配方未随 [openpi](https://github.com/Physical-Intelligence/openpi) 发布。

## 核心摘录（归纳，非全文）

- 论点：机器人应用仍要自建控制器、数据管线和模型；通才模型（文中点名 π₀、π₀.₅、π₀.₆、π\*₀.₆）应扮演类似 LLM API 的物理智能层。
- Weave（旧金山洗衣店现场视频）：称 π₀.₆ SFT 相对 π₀.₅ SFT 提高自主时间占比；把 Weave 数据放进预训练后，连续漏抓序列减少 42%，整筐干预次数减少 50%。柱状图其余刻度未在正文给出。
- Ultra（客户仓库打包）：一段连续镜头写明自主率 96.4%；称 π₀.₆ 相对 π₀.₅ 提高成功率，伙伴数据进预训练后再提高吞吐（件/小时）。正文给出的定性观察是提示跟随更好、长尾恢复策略更多。
- 数字来自伙伴自己的现场统计，任务、本体和干预定义与 PI 实验室论文不同，不能和 π\*₀.₆ / π₀.₇ 的论文表横比。
- **对 wiki 的映射：** [pi-physical-intelligence-layer](../../wiki/entities/pi-physical-intelligence-layer.md)
