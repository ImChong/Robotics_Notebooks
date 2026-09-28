# Emergence of Human to Robot Transfer in Vision-Language-Action Models（arXiv:2512.22414）

> 来源归档（ingest）

- **标题：** Emergence of Human to Robot Transfer in Vision-Language-Action Models
- **短名：** PI human-to-robot transfer
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2512.22414>
- **会议：** RSS 2026，<https://www.roboticsproceedings.org/rss22/p072.html>
- **项目页：** <https://www.pi.website/research/human_to_robot>
- **机构：** 物理智能（Physical Intelligence）；佐治亚理工学院（Georgia Tech）
- **入库日期：** 2026-09-28
- **一句话说明：** π₀.₅ 预训练足够多样之后，把第一人称人视频当成另一种本体一起微调，人→机器人迁移才会出现。

## 开源状态（步骤 2.5，2026-09-28）

- **确认未开源**：项目页、arXiv 与 RSS 页未列训练代码、人视频或机器人数据。引用里的第三方仓库不是本文实现。

## 核心摘录（面向 wiki 编译）

- 动作用 3D 手部位置，不做人机外观对齐、不把人手生成成夹爪。人视频只在微调出现，预训练仍是机器人数据。
- 博客称四个只出现在人视频里的泛化设置上，加入人数据后性能大约 2 倍。论文写迁移幅度随预训练多样性上升：多样性 0% / 25% 时共训人数据几乎无收益，75% / 100% 以及跨本体混合后收益变大。
- 论文还写：Sort Eggs 与 Dresser 上，人数据微调接近目标机器人域内数据；Bussing 上目标机器人数据仍然更有效。
- **对 wiki 的映射：** [paper-pi-human-to-robot](../../wiki/entities/paper-pi-human-to-robot.md)
