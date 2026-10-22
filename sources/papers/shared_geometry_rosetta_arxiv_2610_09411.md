# Shared Geometry as a Rosetta Stone: Cross-Modal Alignment Without Paired Data

- **类型：** paper / cross-modal alignment / representation geometry
- **arXiv：** [2610.09411](https://arxiv.org/abs/2610.09411)（v1：2026-10-07；v2：2026-10-08）
- **PDF：** <https://arxiv.org/pdf/2610.09411>
- **作者：** Dominik Schnaus、Thomas Dagès、Daniel Cremers、Xi Wang、Phillip Isola
- **机构：** TU Munich、MCML、Ulm University、ETH Zurich、MIT
- **项目页：** <https://dominik-schnaus.github.io/unpaired-rosetta/>
- **代码：** <https://github.com/dominik-schnaus/unpaired-rosetta>
- **一句话说明：** 研究独立训练的不同模态编码器能否仅凭嵌入集合各自的几何结构，在没有配对训练样本时学习一个粗粒度跨模态映射。

## 摘录要点

1. **问题与结论：** 常见跨模态对齐依赖配对图文数据。论文证明在表示空间共享足够几何结构时，可以不看配对样本，估计单个正交映射实现粗对齐。
   - **对 wiki 的映射：** [论文实体页](../../wiki/entities/paper-shared-geometry-rosetta.md)；[多模态基础概念](../../wiki/concepts/multimodality-basics.md)

2. **方法：** 先分别聚类两种嵌入并匹配簇几何，随后以 Gromov-Wasserstein 读出粗对应，再交替执行样本匹配与正交 Procrustes 更新映射。
   - **对 wiki 的映射：** [论文实体页](../../wiki/entities/paper-shared-geometry-rosetta.md)；[视觉–语言特征融合](../../wiki/concepts/vision-language-feature-fusion.md)

3. **结果读法：** 作者以 FOSCTTM 评估配对检索质量，并发现 CKA 与无配对对齐质量相关；项目页报告视觉—语言 84 种模型/数据组合中 CKA 对质量变化的线性解释度为 R²=0.73。这个 CKA 分析需要少量配对样本，不是训练对齐器的输入。
   - **对 wiki 的映射：** [论文实体页](../../wiki/entities/paper-shared-geometry-rosetta.md)；[视觉–语言特征融合](../../wiki/concepts/vision-language-feature-fusion.md)

## 参考来源

- [arXiv 摘要与版本记录](https://arxiv.org/abs/2610.09411)
- [作者项目页与交互演示](https://dominik-schnaus.github.io/unpaired-rosetta/)
- [作者官方代码仓库](https://github.com/dominik-schnaus/unpaired-rosetta)
