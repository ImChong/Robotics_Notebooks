# EgoWorld 项目页（ICLR 2026）

> 来源归档（ingest）

- **标题：** EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations
- **类型：** site（官方项目页）
- **发布方：** Junho Park / Andrew Sangwoo Ye / Taein Kwon（LG Electronics · KAIST · Oxford VGG）
- **原始链接：** <https://redorangeyellowy.github.io/EgoWorld/>
- **论文：** <https://arxiv.org/abs/2506.17896>
- **代码：** <https://github.com/redorangeyellowy/EgoWorld>
- **OpenReview：** <https://openreview.net/forum?id=wcTuZG9P2o>
- **入库日期：** 2026-07-24
- **一句话说明：** ICLR 2026 官方落地页：TL;DR、两阶段方法示意、H2O/TACO/Assembly101/Ego-Exo4D 主表、条件模态/骨架/姿态建模消融与失败案例，以及 BibTeX。

## 项目页 / 源码开放核查（步骤 2.5 · 2026-07-24）

| 核查项 | 结论 |
|--------|------|
| **项目页 Code / Resources** | 明确链到 GitHub `redorangeyellowy/EgoWorld` 与 arXiv |
| **开放程度** | **已开源**：训练 `train.py`、评测 `test.py`、MIT 许可；README 提供 H2O 数据、SD-inpainting ckpt、H2O 预训练 ckpt 下载 |
| **数据开放** | 评测依赖公开数据集（H2O 等）；仓库内提供 H2O action inpainting 的 train/test JSON 与 Drive 预处理包链接 |
| **与同名数据集** | **无关** StellarNex [EgoWorld-100W](../blogs/stellarnex_egoworld_100w.md) |

## 摘录要点（与论文分工）

- **对外叙事：** 单张 exo 图 → 稀疏 egocentric RGB / 3D 手姿 / 文本 → 扩散重建稠密 ego 图。
- **主结果口径：** 四基准 + 未见物体/动作/场景/主体；野外手机样例仍优于基线。
- **分析卡片：** 条件模态、LDM vs MAE/MAT、手姿建模策略、生成一致性、错文本可控性、失败案例。

## 对 wiki 的映射

- [EgoWorld（论文）](../../wiki/entities/paper-egoworld.md)
- 姊妹归档：[论文摘录](../papers/egoworld_arxiv_2506_17896.md)、[代码仓](../repos/egoworld.md)
