# Segment Anything（SAM, arXiv:2304.02643）

> 来源归档（ingest · 原文通读）

- **标题：** Segment Anything
- **作者：** Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick（Meta AI Research, FAIR）
- **arXiv：** <https://arxiv.org/abs/2304.02643>
- **通读：** 已完成
- **核对范围：** 已通读 `arXiv:2304.02643v1` 全文（167,093 字符，30 页含图例与标注规范），核对可提示分割任务、MAE ViT 图像编码器 / 提示编码器 / 轻量掩码解码器三段结构，以及 SA-1B 11M 图像 + 1B+ 掩码三阶段数据引擎与零样本评测。

## 核心摘要

- 提出「可提示分割（promptable segmentation）」预训练任务、实时掩码解码架构 SAM 与 11M 图像 / 超 10 亿掩码 SA-1B 数据引擎，确立视觉基础模型从封闭类别检测分割转向提示驱动零样本迁移的标准工程模板。

## 对 wiki 的映射

- [`wiki/concepts/visual-foundation-model-trends.md`](../../wiki/concepts/visual-foundation-model-trends.md)
