# Glob3R 项目页（junyuandeng.github.io/Glob3r）

> 来源归档

- **标题：** Glob3R — Global Structure-from-Motion with 3D Foundation Models
- **类型：** site / project-page
- **URL：** <https://junyuandeng.github.io/Glob3r/>
- **论文：** <https://arxiv.org/abs/2607.09225> — 归档见 [`sources/papers/glob3r_arxiv_2607_09225.md`](../papers/glob3r_arxiv_2607_09225.md)
- **代码：** <https://github.com/aigc3d/Glob3R> — 归档见 [`sources/repos/glob3r.md`](../repos/glob3r.md)
- **机构：** 香港科技大学（HKUST）× 阿里巴巴通义实验室（Tongyi Lab）× 南京大学（NJU）× 复旦大学（Fudan）
- **入库日期：** 2026-07-21
- **一句话说明：** Glob3R 官方项目页：大场景/紧凑场景重建、优化过程与 novel-view synthesis 演示，以及「稠密 tracks → 全局关联 → SfM 精炼」管线叙事。

## 公开信息要点（截至 2026-07-21 核查）

| 项 | 状态 |
|----|------|
| **arXiv / PDF** | 已挂：<https://arxiv.org/abs/2607.09225> |
| **Paper 按钮** | 指向 arXiv abs |
| **Code 按钮** | 已链官方仓 <https://github.com/aigc3d/Glob3R> |
| **数据集 / 权重** | 项目页 **未** 列 Hugging Face / Zenodo 等下载 |
| **开源结论** | **部分开源（占位仓）** — 有 GitHub URL，但仓内仅 README，Inference/Evaluation 仍 TODO |

### 页面内容摘要

- **定位文案：** Global SfM-style reconstruction；把前馈几何预测转成可优化多视图约束。
- **演示区：** Large-scene / Compact-scene 重建、Optimization process、Novel-view synthesis 视频。
- **Pipeline：** 有序序列或检索伪序列 → 窗内局部几何先验与 dense warps → 可靠 tracks 合并为全局关联图 → rotation/translation averaging、BA、稠密重建。
- **BibTeX：** `@article{glob3r2026, ... arXiv:2607.09225}`。

## 为何值得保留

- 步骤 2.5 项目页核查主入口：锁定 **Code URL 已挂但推理未开放** 的边界，避免 wiki 误写「可复现」。
- 提供比 PDF 更直观的定性演示与管线一句话，便于与 LingBot-Map / COLMAP 选型对照。
- 推理代码一旦放出，可在此页与 `sources/repos/glob3r.md` 增量更新。

## 关联资料

- 论文摘录：[`sources/papers/glob3r_arxiv_2607_09225.md`](../papers/glob3r_arxiv_2607_09225.md)
- 代码仓：[`sources/repos/glob3r.md`](../repos/glob3r.md)
- Wiki 实体：[`wiki/entities/paper-glob3r.md`](../../wiki/entities/paper-glob3r.md)
