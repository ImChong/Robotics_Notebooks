# Ego4D（项目页 · ego4d-data.org）

> 来源归档（ingest）

- **标题：** Egocentric 4D Perception (EGO4D)
- **类型：** site / dataset / benchmark consortium
- **官方入口：** <https://ego4d-data.org/>
- **文档 / Start Here：** <https://ego4d-data.org/docs/start-here/>
- **论文：** <https://arxiv.org/abs/2110.07058>
- **代码 / CLI：** <https://github.com/facebookresearch/Ego4D>
- **可视化（需 license）：** <https://visualize.ego4d-data.org/>
- **License 签署：** <https://ego4d.dev/>（审批约 48h；凭证约 14 天，可续期）
- **机构 / 联盟：** Meta（FAIR）牵头 + 13 所大学等国际联盟（项目页 Team 区）
- **入库日期：** 2026-08-08
- **一句话说明：** Ego4D 官方站点：展示约 **3,670** 小时全球 egocentric 日常视频规模、五大 benchmark 挑战、隐私/伦理说明，并提供文档、license、可视化与下载入口。

## 开源与数据开放核查（步骤 2.5 · 入库日 2026-08-08）

| 项 | 状态 | 入口 |
|----|------|------|
| **项目页** | **已公开** | <https://ego4d-data.org/> |
| **论文** | **已公开** | arXiv:2110.07058 |
| **数据集 / 标注** | **受控开放** | 签署 Ego4D license → AWS 凭证 → CLI 下载；**非** ungated 全量镜像 |
| **代码** | **已开源（MIT）** | [facebookresearch/Ego4D](https://github.com/facebookresearch/Ego4D)（`pip install ego4d`） |
| **预计算特征** | **随数据提供** | 文档 / CLI；SlowFast 等 action features |
| **Benchmark 基线仓** | **部分 / 分散** | [EGO4D GitHub org](https://github.com/EGO4D/) + docs |

**结论：** 数据 **已发布但需 license**；下载/特征/可视化工具链 **已开源**。勿写成「无代码」或「匿名无门槛全量下载」。

## 页面摘录要点

- **规模叙事：** 3,670 小时；923–931 佩戴者（站点与论文数字口径略异，以数据卡/论文表为准）；74 地点、9 国；相对既往 egocentric 集约 **20×** 小时量级。
- **多样性：** 地理、场景、参与者与模态（3D 扫描、音频、gaze、stereo、多机同步、叙述）。
- **五大 Challenges：** Episodic Memory · Hand–Object Interaction · AV Diarization · Social · Forecasting。
- **隐私：** 各伙伴机构政策 + 去标识管线；问题联系 `privacy@ego4d-data.org`；一般咨询 `info@ego4d-data.org`。
- **引用：** Grauman et al., *Ego4D: Around the World in 3,000 Hours of Egocentric Video*, arXiv:2110.07058（CVPR 2022）。
- **工程提醒（Start Here）：** 全量视频量级约 **数 TB–数十 TB**；优先按 benchmark / 模态子集下载；可先 `--datasets viz` 浏览。

## 对 wiki 的映射

- [Ego4D 论文实体](../../wiki/entities/paper-ego4d.md) — 主升格
- [Ego 分类 01：数据采集](../../wiki/overview/ego-category-01-data-collection.md)
- [HumanNet Table 1](../../wiki/comparisons/humannet-table1-human-video-corpora.md)
- [EgoVerse](../../wiki/entities/paper-egoverse.md) — 后续操纵向 / 联盟式人示教对照
- [RekaDaily-10k](../../wiki/entities/rekadaily-10k-dataset.md) — 家务 ego 开放语料对照

## 交叉链接（sources 互指）

- 论文：[ego4d_arxiv_2110_07058.md](../papers/ego4d_arxiv_2110_07058.md)
- 仓库：[ego4d.md](../repos/ego4d.md)
