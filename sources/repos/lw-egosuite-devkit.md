# LW-Egosuite-DevKit

> 来源归档（国内具身开源全景）

- **标题：** LW-Egosuite-DevKit
- **类型：** repo
- **机构：** 光轮智能
- **链接：** https://github.com/LightwheelAI/LW-Egosuite-DevKit
- **分类：** 数据集/Benchmark
- **入库日期：** 2026-09-06
- **一句话说明：** 光轮智能 开源 MCAP 转换与可视化工具链，服务 [EgoSuite-Open100K](../sites/egosuite-open100k-lightwheel.md) 数据质检与管线接入。
- **关联数据：** [EgoSuite-Open100K](../sites/egosuite-open100k-lightwheel.md) · [EgoStandard](../datasets/lightwheel-egostandard.md) · [EgoPro](../datasets/lightwheel-egopro.md)
- **沉淀到 wiki：** [`wiki/entities/cn-os-lw-egosuite-devkit.md`](../../wiki/entities/cn-os-lw-egosuite-devkit.md) · [`wiki/entities/egosuite-open100k.md`](../../wiki/entities/egosuite-open100k.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## README / 包核查（2026-10-10）

- **README：** `raw.githubusercontent.com/LightwheelAI/LW-Egosuite-DevKit/main/README.md` 可读（200）；GitHub API 经代理 403。
- **许可：** Apache-2.0（Copyright 2026 Lightwheel Team）。
- **PyPI：** `lw-egosuite-devkit`，0.1.2（2026-03-03）→ 1.0.2（2026-08-06）。
- **功能：** `lw-egosuite convert`（原始 MCAP → `_vis.mcap`，骨架/轨迹/语义叠加）、LW-VIZ（<https://foxviz.lightwheel.net/>）可视化、`lw-egosuite export-video`（MP4 导出）、Python `iter_messages` / `iter_video_frames`。
- **数据文档：** <https://docs.lightwheel.net/egocentric_data/>（MCAP topic 与 LeRobot v3 导出规范）。
- **母产品：** [Lightwheel EgoSuite 归档](../blogs/lightwheel_egosuite.md)

## 对 wiki 的映射

- [wiki/entities/cn-os-lw-egosuite-devkit.md](../../wiki/entities/cn-os-lw-egosuite-devkit.md)
- [wiki/entities/lightwheel-egosuite.md](../../wiki/entities/lightwheel-egosuite.md)
