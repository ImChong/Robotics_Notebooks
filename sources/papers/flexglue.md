# FlexGlue：稀疏到稠密特征匹配与视觉消歧

> 来源归档（paper）

- **标题：** FlexGlue: Sparse-to-Dense Feature Matching and Visual Disambiguation
- **作者：** Philipp Lindenberger、Paul-Edouard Sarlin、Ondrej Miksik、Marc Pollefeys
- **机构：** 苏黎世联邦理工、谷歌、微软 Spatial AI Lab
- **版本：** [arXiv:2610.32976v1](https://arxiv.org/abs/2610.32976v1)，2026-10-23
- **代码：** https://github.com/cvg/FlexGlue
- **入库日期：** 2026-11-06

## 核验范围

已核对论文方法、实验表格及局限章节；全文及附录尚未逐段通读，待补全。本记录和实体页暂供草稿审阅。

## 核心内容

- 用冻结的 DINOv3 ConvNeXt-B 提取两图稠密特征，在 stride 8 网格上进行自注意力和交叉注意力。
- 查询点与目标图网格相关，局部窗口读出及轻量修正头预测对应坐标；逐层共视性分数可用于提前拒绝相似但不重叠的图像。
- LaMAR-DG 含三个场景，每个场景 500 对正例和 500 对困难负例；ETH3D 与 InLoc 数值的协议和限制见实体页。

## 对 wiki 的映射

- [FlexGlue 实体](../../wiki/entities/paper-flexglue.md)
- [官方仓库归档](../repos/flexglue.md)
