# deworldsg2026.github.io（DeWorldSG 项目页）

- **标题：** DeWorldSG — Depth-Aware 3D Semantic Scene Graph Generation via World-Model Priors
- **类型：** site / project-page
- **URL：** <https://deworldsg2026.github.io/>
- **配套论文：** [DeWorldSG（arXiv:2607.00889）](https://arxiv.org/abs/2607.00889) — 归档见 [`sources/papers/deworldsg_arxiv_2607_00889.md`](../papers/deworldsg_arxiv_2607_00889.md)
- **会议：** ECCV 2026
- **静态站仓库：** <https://github.com/deworldsg2026/deworldsg2026.github.io>（仅项目页源码，非算法仓）
- **代码（截至 2026-09-11）：** 页上 **Coming Soon**，无官方训练/推理 GitHub 或 Hugging Face 链接
- **入库日期：** 2026-09-11

## 一句话摘要

KAIST × TUM × MCML 的 **DeWorldSG** 官方站点：展示 RGB-D → 深度感知 3D 高斯节点 → 增量全局场景图 → V-JEPA 2 关系 refine 的方法图与 3DSSG/ReplicaSSG 结果，附 Room/Office/Apartment demo 视频。

## 公开信息要点（截至入库日）

- **机构：** KAIST；Technical University of Munich；Munich Center for Machine Learning (MCML)。通讯：Benjamin Busam、Woontack Woo。
- **TL;DR：** 深度感知概率物体建模 + 世界模型引导关系推理 → 时空一致 3D 语义场景图。
- **方法图：** 2D SSG → SAM mask + Dual-Domain Depth Refinement → 3D 高斯 lifting → 语义/高斯 merge → V-JEPA 2 关系 refine。
- **结果区：** 相对 prior SoTA，triplet recall **+77.4%**、predicate recall **+23.2%**（摘要口径）。
- **Demo：** Room / Office / Apartment 三组场景视频嵌入。
- **步骤 2.5 结论：** 论文写 open-sourced，但页上 Code 为 **Coming Soon**；GitHub 仅见 pages 仓 — **待发布**，勿误标为已可复现。

## 为何值得保留

- **非 PDF 证据：** 方法 pipeline 图与 demo 视频比表格更直观展示 incremental merge 与关系 refine。
- **开源边界以页上实际链接为准：** 与 arXiv 摘要「open-sourced」存在落差，便于后续 lint 跟进。

## 关联资料

- 论文归档：[`sources/papers/deworldsg_arxiv_2607_00889.md`](../papers/deworldsg_arxiv_2607_00889.md)
- 升格：[`wiki/entities/paper-deworldsg.md`](../../wiki/entities/paper-deworldsg.md)
