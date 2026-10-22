# unpaired-rosetta（Dominik Schnaus）

- **类型：** repo
- **来源：** 论文作者官方仓库
- **链接：** <https://github.com/dominik-schnaus/unpaired-rosetta>
- **项目页：** <https://dominik-schnaus.github.io/unpaired-rosetta/>
- **关联论文：** [arXiv:2610.09411](https://arxiv.org/abs/2610.09411)
- **许可证：** MIT
- **语言 / 框架：** Python；仓库 README 说明环境测试为 Python 3.14、PyTorch 2.14，并用 pixi 管理环境
- **开源状态：** 已开源；包含对齐库、实验脚本、测试与 Notebook。代码不含项目自创训练数据集。
- **入库日期：** 2026-10-22
- **一句话说明：** 跨模态无配对对齐方法的作者实现，可直接接收两组未配对嵌入，并可选加入少量已知配对。

## README 要点（编译自上游）

- **API：** WassersteinProcrustes.fit(images, captions) 接收两组不要求逐样本配对的嵌入；可选参数用于传入已知配对。提供 similarity() 和 transform()。
- **算法实验：** experiments/ 按论文实验组织；README 列出视觉—语言、多数据集、跨模态、共享几何、少样本对齐、文本到图像等实验入口。
- **Notebook：** unpaired_rosetta.ipynb 展示选择数据集与编码器、计算或载入嵌入、运行对齐与验证的流程。
- **复现边界：** 需按 pixi 环境安装依赖并准备或下载相应嵌入和数据；部分模态实验需要额外环境与外部数据。仓库测试覆盖评估及 Notebook/库映射等一致性。

## 对 wiki 的映射

- [统一论文实体页](../../wiki/entities/paper-shared-geometry-rosetta.md)
- [作者项目页归档](../sites/shared-geometry-rosetta.md)
- [论文来源归档](../papers/shared_geometry_rosetta_arxiv_2610_09411.md)
