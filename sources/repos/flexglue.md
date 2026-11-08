# FlexGlue 官方仓库

> 来源归档（repo）

- **代码：** https://github.com/cvg/FlexGlue
- **论文：** https://arxiv.org/abs/2610.32976v1
- **官方 COLMAP 集成：** [COLMAP 文档和实现归档](../sites/flexglue-colmap.md)
- **核对提交：** c6cbc7a933bb70ab58608e8835d99b6cc4adca1a
- **入库日期：** 2026-11-06
- **实体：** [FlexGlue](../../wiki/entities/paper-flexglue.md)

## 开放范围与入口

README 提供 PyTorch 推理入口 `FlexGlue().match` 和 `match_pair`，支持 ONNX 导出；官方仓库提供推理代码与权重。训练及评测代码的发布状态应以官方仓库清单为准。代码为 MIT；权重中 DINOv3 ConvNeXt-B 主干适用 DINOv3 许可，使用前须单独核查其限制。

源码核对：`flexglue/flexglue.py` 的 `match_pair`、`match`、`describe_inputs`、`decode_pair`、`keypoint_correspondences`；`flexglue/descriptor.py`。运行尚未在本环境验证。

## ONNX 与 COLMAP 路径核对（2026-11-08）

官方 README 已给出 v0.1 的 `flexglue_descriptor.onnx` 与稀疏匹配 `flexglue_matcher{,_fp16}.onnx`，另有适合 COLMAP 的 early-stop 变体；`onnx/match_pair.py` 以 SIFT 关键点、ONNX Runtime 描述器与匹配器完成独立推理。COLMAP `src/colmap/feature/flexglue.cc` 在匹配时读取图像像素、缓存每张图的稠密 ONNX 特征，再将关键点与两张图的 dense features 喂给匹配器。COLMAP 的 `min_covisibility` 默认 0.5，PyTorch 默认 0.0，应用时不要混淆。

来源：
- https://github.com/cvg/FlexGlue/blob/main/onnx/match_pair.py
- https://github.com/colmap/colmap/blob/main/src/colmap/feature/flexglue.cc
- https://colmap.github.io/features.html#flexglue

官方仓库 README 的 Checklist 当前仍列训练和评测代码将后续发布；不能把可推理/可集成误写成完整训练复现已开源。
