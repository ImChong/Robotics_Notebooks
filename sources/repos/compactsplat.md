# hebing-sjtu/CompactSplat

> 来源归档（官方代码仓）

- **标题：** CompactSplat: Spatially Adaptive Gaussian Distribution for Feedforward Scene Reconstruction
- **类型：** repo
- **组织：** 上海交通大学研究团队
- **代码：** <https://github.com/hebing-sjtu/CompactSplat>
- **项目页：** <https://hebing-sjtu.github.io/CompactSplat-website/>
- **论文：** NeurIPS 2026 Poster；arXiv:2610.13177 — [论文摘录](../papers/compactsplat_arxiv_2610_13177.md)
- **权重：** <https://huggingface.co/binghe-sjtu/compactsplat>
- **许可证：** MIT（以仓库许可证文件为准）
- **入库日期：** 2026-10-23
- **一句话说明：** 官方实现包含自适应图像分区、层级高斯属性采样、配置、训练与评估脚本；示例针对 RealEstate10K 等场景重建数据集。

## 开源核查（2026-10-23）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开） |
| 代码入口 | `python -m src.main`；仓内有训练 / 评估脚本与配置 |
| 权重 | Hugging Face 仓库已提供 CompactSplat 权重入口 |
| 数据 | 需自行准备并预处理 RealEstate10K、DL3DV、ScanNet 等数据；遵守各数据集许可 |
| 复现注意 | README 指出现有分区判据和默认阈值尚不同于论文设定；完整复现前核对 `docs/REPRODUCIBILITY.md` |

## 对 wiki 的映射

- 项目页归档：[compactsplat-project-page.md](../sites/compactsplat-project-page.md)
- 论文归档：[compactsplat_arxiv_2610_13177.md](../papers/compactsplat_arxiv_2610_13177.md)
- 论文实体：[paper-compactsplat.md](../../wiki/entities/paper-compactsplat.md)
