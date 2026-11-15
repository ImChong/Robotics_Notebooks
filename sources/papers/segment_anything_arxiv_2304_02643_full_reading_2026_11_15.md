# Segment Anything：原文通读与实现复核（2026-11-15）

- **类型：** paper
- **原文：** <https://arxiv.org/html/2304.02643>
- **版本：** 2026-11-15 获取的 HTML；具体正文、附录和代码范围如下。
- **代码：** <https://github.com/facebookresearch/segment-anything>
- **核对日期：** 2026-11-15
- **核对范围：** arXiv:2304.02643 HTML（2026-11-15 获取），正文 §1–8 与附录 A–G（架构与训练、自动 mask 生成、RAI、实验协议、人工评审、数据/模型卡与标注说明）；另读官方 `segment_anything/predictor.py` 的图像缓存/提示入口及 `modeling/sam.py` 全文。参考文献浏览，未训练或运行模型。
- **通读：** 已完成

本次复核单独存档；[历史摘录](segment_anything_arxiv_2304_02643.md)保留原有日期和发布状态，不能代替下述版本及协议核对。

## 原文与实现核对

- [原文与附录](https://arxiv.org/html/2304.02643)把图像编码、提示编码和双向轻量解码器分开；三 mask 输出处理单点歧义，IoU head 用于排序，三档不是固定语义部件树。
- §7.1 的 23 数据集单点实验中，按模型自身置信度选 mask 时在 16 个数据集优于 RITM；按真值选最佳 mask 的 oracle 协议不同，不应报告为默认部署成绩。
- §7.5 与附录 D.5 的 text-to-mask 是额外训练的探索模型：以 CLIP 图像特征作训练提示、文本特征作推理提示；不能从该实验推断公开 `SamPredictor.predict` 原生支持任意文本。
- 约 50 ms 指缓存图像特征后的提示编码/解码，不包含重型图像编码。§8 同时说明细结构、小孤立区域及多次交互的高 IoU 目标仍有限制。
- [SamPredictor](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py) 的 `set_image/set_torch_image` 调整尺寸并缓存编码；`predict` 变换点/框坐标后调用 `predict_torch`；[Sam](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/sam.py) 的 `postprocess_masks` 去 padding 并恢复原图分辨率。公开接口是几何提示和低分辨率 mask，输出并不附带类别或 3D 坐标。
- SA-1B 只发布自动标注的无类别 mask，研究数据许可与 Apache-2.0 模型代码分别处理。

## 对 wiki 的映射

- [Segment Anything 详情](../../wiki/entities/paper-segment-anything.md)
