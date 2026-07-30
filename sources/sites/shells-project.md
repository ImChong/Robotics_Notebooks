# SHELLS 项目页（syntec-research.github.io/SHELLS）

> 来源归档（ingest 配套站点）

- **URL：** <https://syntec-research.github.io/SHELLS/>
- **对应论文：** [Topologically Consistent Multi-view 3D Head Reconstruction via Coarse-Guided Layered Surface Sampling](https://arxiv.org/abs/2605.31283)（arXiv:2605.31283，Google，SIGGRAPH 2026）
- **入库日期：** 2026-07-30
- **一句话说明：** 官方落地页：架构图、性能/遮挡/少视角演示、与 TEMPEH 等对照叙事、BibTeX；**无代码/权重链接**。
- **代码：** 截至 2026-07-30 **未列出** GitHub / Hugging Face / Zenodo 等入口（仅 arXiv、本地 PDF、BibTeX）。

## 页面要点（2026-07 快照）

### 核心主张（TL;DR）

From calibrated multi-view images, SHELLS reconstructs **18k-vertex** 3D heads in **0.08 seconds**. It aggregates DINOv2 features via projective surface-aware feature sampling, allowing a transformer to predict dense semantic meshes **3.5× faster** with **88% less GPU memory** than volumetric SOTA. Supports few-view (down to 2) and implicit occlusion completion.

### 展示块

| 区块 | 内容 |
|------|------|
| Architecture | DINOv2+LoRA → 稀疏图粗预测 → 法向分层壳 → 共享 XCiT 精预测 |
| Performance registration | 逐帧表情表演配准，时序较平滑 |
| Implicit occlusion handling | 口腔内侧等遮挡区靠全局注意力补全 |
| Robustness to #views | 2 / 3 / 4 / 10 视角仍可合理重建 |

### BibTeX（项目页）

```bibtex
@inproceedings{Bolkart2026SHELLS,
  author    = {Bolkart, Timo and Wang, Daoye and Chandran, Prashanth},
  title     = {Topologically Consistent Multi-view 3D Head Reconstruction via Coarse-Guided Layered Surface Sampling},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  keywords  = {Registration, 3D Head Reconstruction},
  series    = {SIGGRAPH Conference Papers '26}
}
```

## 开源核查（步骤 2.5）

| 项 | 状态（2026-07-30） |
|----|-------------------|
| 项目页 Code / Resources | **无** 仓库链接 |
| 论文 Code availability | **未给出** 公开 URL |
| 权重 / 数据集 | **未公开**（合成数据为内部流程描述） |
| 结论 | **未开源**；后续若发布需补 `sources/repos/` 与 wiki「源码运行时序图」 |

## 对 wiki 的映射

- 与 [sources/papers/shells_arxiv_2605_31283.md](../papers/shells_arxiv_2605_31283.md) 配对
- 实体页：[wiki/entities/paper-shells-layered-surface-sampling.md](../../wiki/entities/paper-shells-layered-surface-sampling.md)
