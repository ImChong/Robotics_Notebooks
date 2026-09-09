# NBS — Intrinsic Experimental 项目页

> 来源归档（ingest · 步骤 2.5 · 2026-09-09）

- **标题：** NBS: No Bias Stereo
- **类型：** site（Intrinsic Experimental 官方项目页）
- **发布方：** Intrinsic（Google）× Texas A&M
- **原始链接：** <https://intrinsic-experimental.github.io/nbs-website/>
- **论文：** <https://arxiv.org/abs/2608.28933>（页内 Paper / BibTeX）
- **入库日期：** 2026-09-09
- **一句话说明：** TL;DR「A plain ViT is all stereo needs」；交互式与 GT / Selective-IGEV / FoundationStereo / S²M² 对比滑块；ETH3D + SimpleProc + XYZ-IBD 全表 SOTA 与 A100 效率表。

## 步骤 2.5 开源核查（2026-09-09）

| 项 | 项目页 |
|----|--------|
| **论文** | arXiv [2608.28933](https://arxiv.org/abs/2608.28933) |
| **GitHub** | **未列链接** — 截至入库日 **无官方代码仓** |
| **视频/演示** | 交互 disparity 对比、3D 旋转重建 |
| **数据** | 提及 2.4M internal synthetic + 13 public；**无下载入口** |

## 主页摘录

- **方法：** 单端到端 ViT；patchify 校正双目 → 局部/全局交替 self-attention → DPT 视差解码；**无** frozen depth expert、**无** correlation volume、**无** iterative refinement。
- **定性：** 容器空洞、杂乱物体、烤架细结构等 prior 方法缺失区域 NBS 更完整。
- **效率（SimpleProc-S 966×546, A100 FP16）：** NBS **0.060 s / 1.23 GB** vs FoundationStereo **0.872 s / 6.74 GB** vs S²M² **0.240 s / 3.52 GB**（精度仍优于或可比）。

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-nbs-no-bias-stereo.md`](../../wiki/entities/paper-nbs-no-bias-stereo.md)
- 生态对照：[`wiki/methods/stereo-matching-foundation-models.md`](../../wiki/methods/stereo-matching-foundation-models.md)
- 论文摘录：[`sources/papers/nbs_arxiv_2608_28933.md`](../papers/nbs_arxiv_2608_28933.md)
