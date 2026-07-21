# EgoHTR 项目页（egohtr.github.io）

> 来源归档

- **标题：** EgoHTR — Egocentric 4D Demonstrations of Human Terrain Traversal
- **类型：** site / project-page
- **URL：** <https://egohtr.github.io>
- **论文：** <https://arxiv.org/abs/2607.13472> — 归档见 [`sources/papers/egohtr_arxiv_2607_13472.md`](../papers/egohtr_arxiv_2607_13472.md)
- **机构：** 苏黎世联邦理工（ETH Zürich）× 斯坦福大学（Stanford）× 加州大学伯克利分校（UC Berkeley）× 慕尼黑工业大学（TU Munich）
- **入库日期：** 2026-07-21
- **一句话说明：** EgoHTR 官方项目页：数据集统计/场景浏览、采集传感器表、Human2Robot 重定向说明，以及 G1 感知全身控制与 HMR 基准应用叙事。

## 公开信息要点（截至 2026-07-21 核查）

| 项 | 状态 |
|----|------|
| **arXiv / PDF** | 已挂：<https://arxiv.org/abs/2607.13472> |
| **Dataset** | 页头按钮文案 **Dataset (coming soon)**；**无** Hugging Face / Zenodo / 下载 URL |
| **Code** | 页头按钮文案 **Code (coming soon)**；**无** 可点击训练/重建仓库 URL |
| **GitHub org** | <https://github.com/egohtr> 目前仅 [`egohtr/egohtr.github.io`](https://github.com/egohtr/egohtr.github.io)（项目站源码），**非** 复现仓 |
| **开源结论** | **宣称将开源 / 待发布**（论文写 open-source pipeline，项目页尚未列有效数据/代码链接） |

### 页面内容摘要

- **规模卡片：** 7 scenes / 55 sequences / 1.37 h / ~150k frames @ 30 fps / 8 subjects（4F/4M）/ 36 multi-view / 0.7 h mocap GT。
- **场景：** Debris Field、Lab Hall、Gym Hall、Office；动作标签含 Stairs、Parkour、Climbing、Flips、Sitting/Lay 等。
- **采集表：** Aria Gen.1（必选）、Rokoko Pro II（必选）、Leica BLK2GO（必选）、第二副 Aria / 固定相机（可选）。
- **Human2Robot：** 场景感知 retarget 基于 [OmniRetarget](https://arxiv.org/abs/2509.26633)、[GMR](https://arxiv.org/abs/2510.02252)、[CoACD](https://arxiv.org/abs/2205.02961)。
- **应用：** Perceptive whole-body control（Unitree G1）；reconstruction / estimation benchmarking。

## 为何值得保留

- 步骤 2.5 项目页核查主入口：用「coming soon」文案锁定开放边界，避免 wiki 误写「已可下载」。
- 提供比 PDF 更易浏览的序列/模态表，便于与 AMASS / SLOPER4D 等选型对照。
- 后续数据/代码一旦放出，可在此页与 `sources/repos/` 增量更新。

## 关联资料

- 论文摘录：[`sources/papers/egohtr_arxiv_2607_13472.md`](../papers/egohtr_arxiv_2607_13472.md)
- Wiki 实体：[`wiki/entities/paper-egohtr.md`](../../wiki/entities/paper-egohtr.md)
