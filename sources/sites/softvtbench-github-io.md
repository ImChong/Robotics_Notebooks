# softvtbench.github.io（SoftVTBench 项目页）

- **标题：** SoftVTBench — A Safety-Aware Visuo-Tactile Benchmark for Deformable Object Manipulation
- **类型：** site / project-page
- **URL：** <https://softvtbench.github.io/>
- **入库日期：** 2026-07-29
- **配套论文：** [SoftVTBench（arXiv:2607.04234）](https://arxiv.org/abs/2607.04234) — 归档见 [`sources/papers/softvtbench_arxiv_2607_04234.md`](../papers/softvtbench_arxiv_2607_04234.md)
- **代码：** <https://github.com/TuojingAI/SoftVTBench>
- **数据集镜像：** [Hugging Face](https://huggingface.co/datasets/Arthur12137/SoftVTBench) · [ModelScope](https://www.modelscope.cn/datasets/Arthur12137/SoftVTBench)（以 README 为准；页头 Dataset 按钮截至入库日仍标 coming soon）

## 一句话摘要

官方项目页展示 **四套件视触觉评测**、Goal vs Safety Success 叙事，以及 π₀.₅ Vision / Visuo-Tactile 主结果表与形变分布；强调「过松掉落、过紧过压」之间的安全交互包络。

## 公开信息要点（截至入库日）

- **规模标语：** 4 Task Suites · 2,000 Episodes（论文口径；公开托管约 1,628）· 33 Assets · Vision + Tactile RGB + Marker Motion + Proprioception。
- **套件：** Object-Rigid / Spatial-Rigid（LIBERO 风格刚体对照）与 Object-Soft / Spatial-Soft（可变形抓放）。
- **核心公式叙事：** Safety Success = Goal Success × Safe Interaction（无掉落 + 峰值形变 ≤ 标定阈值）。
- **主结果（摘录）：** Object-Soft Safety VO 21.4% → VT 35.6%；Spatial-Soft 32.6% → 44.6%；Goal 在软体上接近。
- **开源边界：** 页上 **GitHub Code** 可进；Paper/Dataset 按钮文案滞后，数据实际由仓库 README 指向 HF/ModelScope。

## 为何值得保留

- **非 PDF 证据：** 套件视频与 teaser 直观展示安全包络与 Goal–Safety gap。
- **复现入口枢纽：** 与 GitHub / arXiv / 数据镜像互链，ingest 步骤 2.5 核查以本页 Code 链为准。

## 关联资料

- 论文归档：[`sources/papers/softvtbench_arxiv_2607_04234.md`](../papers/softvtbench_arxiv_2607_04234.md)
- 代码归档：[`sources/repos/softvtbench.md`](../repos/softvtbench.md)
- wiki：[`wiki/entities/paper-softvtbench.md`](../../wiki/entities/paper-softvtbench.md)
