# GenIA 官方项目页

- **类型：** project site
- **标题：** GenIA: Generative Reconstruction with Test-Time Input Alignment
- **网址：** https://facebookresearch.github.io/GenIA/
- **论文：** https://arxiv.org/abs/2610.12388
- **代码：** https://github.com/facebookresearch/GenIA
- **作者机构：** Tübingen AI Center, University of Tübingen；Meta Reality Labs
- **核查日期：** 2026-10-13
- **开放结论：** **已开源**：项目页 Code 按钮指向公开官方仓库；仓库包含安装脚本、推理 demo 和配置文档。运行需接受 SAM 3D Objects 权重许可并下载模型权重。
- **一句话说明：** 项目页展示 GenIA 在单图、多视图、视频输入上的重建结果，介绍冻结 SAM 3D 先验上的测试时几何/外观对齐、定量比较、消融和运行时间。

## 项目页要点

- 输入覆盖单张图像、静态场景稀疏多视图和单目视频；输出为规范物体高斯/网格、逐帧位姿，以及动态输入下的逐帧形变。
- 官方方法将每个场景分为粗形状、逐帧姿态去噪、对齐外观预测和可选测试时 refinement 四个阶段。
- 项目页明确动态几何依赖 ActionMesh；位姿旋转仍依赖生成先验并由 ICP 修正，因此结果可能受几何、深度、相机和配准误差影响。
- 动态运行时间与 SAM 3D 单图基线相比约 13 倍（含 ActionMesh），而非视频直接 feed-forward；应把质量与较重的每场景推理成本一起评估。

## 对 wiki 的映射

- [GenIA 论文与代码实体页](../../wiki/entities/paper-genia-arxiv-2610-12388.md)

## 交叉归档

- 论文摘录：[genia_arxiv_2610_12388.md](../papers/genia_arxiv_2610_12388.md)
- 官方仓库：[facebookresearch-genia.md](../repos/facebookresearch-genia.md)

## 参考来源

- [GenIA project page](https://facebookresearch.github.io/GenIA/)
- [arXiv:2610.12388](https://arxiv.org/abs/2610.12388)
- [facebookresearch/GenIA](https://github.com/facebookresearch/GenIA)
