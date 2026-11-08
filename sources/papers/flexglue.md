# FlexGlue：稀疏到稠密特征匹配与视觉消歧

> 来源归档（paper）

- **标题：** FlexGlue: Sparse-to-Dense Feature Matching and Visual Disambiguation
- **作者：** Philipp Lindenberger、Paul-Edouard Sarlin、Ondrej Miksik、Marc Pollefeys
- **机构：** 苏黎世联邦理工、谷歌、微软 Spatial AI Lab
- **版本：** [arXiv:2610.32976v1](https://arxiv.org/abs/2610.32976v1)，2026-10-23
- **代码：** https://github.com/cvg/FlexGlue
- **入库日期：** 2026-11-06

## 核验范围

- **核对范围：** 已查阅 arXiv v1 正文第 1–6 节、方法图 1/2、表 1–8、附录 B–G 的主要实验、训练设置与局限，并对 PDF 的模型架构、困难负例、匹配模式及训练损失图表做了视觉核验；尚未逐段完整审阅所有附录图像/表格及参考文献。
- **通读：** 未完成
- **核对日期：** 2026-11-08
- **待办：** 逐段核读全文与全部附录、图 8–14 的所有定性样例，并在完成后更新阅读状态；在此之前不将节点健康度申报为 3/3。

## 一手资料补充

- 官方论文 HTML：https://arxiv.org/html/2610.32976v1
- 官方论文 PDF：https://arxiv.org/pdf/2610.32976v1
- COLMAP 已在当前官方文档中收录 `FLEXGLUE` 匹配器；构建需启用 ONNX，且必须设置 `FlexGlueMatching.image_path`。部署说明与实现证据见 [COLMAP 官方集成归档](../sites/flexglue-colmap.md)。
- 论文第 4.5 节表 4 的 ETH3D SfM 测试（SuperPoint + FlexGlue）中，启用共视过滤后 AUC@5° 为 92.3、启用早停为 91.6；COLMAP 文档的 35% 早停比例是另一组采用默认 `min_covisibility=0.5`、`early_stop_decay=0.5` 的运行设置，不应直接混为同一指标。

## 核心内容

- 用冻结的 DINOv3 ConvNeXt-B 提取两图稠密特征，在 stride 8 网格上进行自注意力和交叉注意力。
- 查询点与目标图网格相关，局部窗口读出及轻量修正头预测对应坐标；逐层共视性分数可用于提前拒绝相似但不重叠的图像。
- LaMAR-DG 含三个场景，每个场景 500 对正例和 500 对困难负例；ETH3D 与 InLoc 数值的协议和限制见实体页。

## 对 wiki 的映射

- [FlexGlue 实体](../../wiki/entities/paper-flexglue.md)
- [官方仓库归档](../repos/flexglue.md)
