# FlexGlue 官方仓库

> 来源归档（repo）

- **代码：** https://github.com/cvg/FlexGlue
- **论文：** https://arxiv.org/abs/2610.32976v1
- **核对提交：** c6cbc7a933bb70ab58608e8835d99b6cc4adca1a
- **入库日期：** 2026-11-06
- **实体：** [FlexGlue](../../wiki/entities/paper-flexglue.md)

## 开放范围与入口

README 提供 PyTorch 推理入口 `FlexGlue().match` 和 `match_pair`，支持 ONNX 导出；官方仓库提供推理代码与权重。训练及评测代码的发布状态应以官方仓库清单为准。代码为 MIT；权重中 DINOv3 ConvNeXt-B 主干适用 DINOv3 许可，使用前须单独核查其限制。

源码核对：`flexglue/flexglue.py` 的 `match_pair`、`match`、`describe_inputs`、`decode_pair`、`keypoint_correspondences`；`flexglue/descriptor.py`。运行尚未在本环境验证。
