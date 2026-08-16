# pz0826.github.io/LEGO-Webpage（LEGO 项目页）

- **标题：** LEGO: Leveled Language Gaussian Splatting
- **类型：** site / project-page
- **URL：** <https://pz0826.github.io/LEGO-Webpage/>
- **配套论文：** [LEGO（arXiv:2608.10057）](https://arxiv.org/abs/2608.10057) — 归档见 [`sources/papers/lego_leveled_language_gs_arxiv_2608_10057.md`](../papers/lego_leveled_language_gs_arxiv_2608_10057.md)
- **代码：** <https://github.com/WHU-USI3DV/LEGO> — 归档见 [`sources/repos/lego.md`](../repos/lego.md)
- **入库日期：** 2026-08-16

## 一句话摘要

武汉大学 × 香港科技大学的 **ECCV 2026** 项目页：把多视角 SAM 粒度重分级成统一 3D 结构层级，再接到 CLIP 与层级语言场景图，做多粒度开放词汇理解与 LLM 空间推理。

## 公开信息要点（截至入库日）

- **机构：** Wuhan University；Hong Kong University of Science and Technology。
- **页首：** ECCV 2026；作者链到个人主页（Yuning Peng / Haiping Wang / Yuan Liu / Zhen Dong 等）。
- **资源按钮：** Paper → arXiv:2608.10057；**Code → `WHU-USI3DV/LEGO`**（不是占位）。实验室入口 [`WHU-USI3DV`](https://github.com/WHU-USI3DV/)。
- **演示：** Room 场景 RGB vs LEGO Agent 视频；Bonsai / Counter / Garden / Kitchen / Teatime / Bulldozer / Truck 等层级分解与 grounding 可视化。
- **方法叙事：** 语义–尺度脱钩（同级大小车必须同一层级）；共视 + 3D 尺度聚类定级；层级对比蒸馏；CLIP grounding；level-wise language scene graph。
- **致谢对照：** GARField、SAGA、[gsplat](https://github.com/nerfstudio-project/gsplat)。同组前作 [GAGS](https://pz0826.github.io/GAGS-Webpage/) 不在本页主链，但是同一作者线。

## 为何值得保留

- **步骤 2.5 的非 PDF 证据：** Code 按钮与仓库 URL 以项目页为准，不能只看 arXiv 摘要。
- **层级 vs 尺度** 的对比图比论文文字更直观，适合对照 [2D→3D 语义提升 Gap](../../wiki/concepts/2d-to-3d-semantic-lifting-gap.md)。
- 与官方仓 README 的 `lego run` / `viewer` / CoR 入口互证。

## 关联资料

- 论文归档：[`sources/papers/lego_leveled_language_gs_arxiv_2608_10057.md`](../papers/lego_leveled_language_gs_arxiv_2608_10057.md)
- 代码仓库：[`sources/repos/lego.md`](../repos/lego.md)
