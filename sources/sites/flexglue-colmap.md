# FlexGlue 在 COLMAP 的官方集成

> 来源归档（site）：COLMAP 官方文档与 C++ 实现，归属 FlexGlue 同一项目；不创建新的 wiki 实体。

- **文档：** https://colmap.github.io/features.html#flexglue
- **COLMAP 官方源码：** https://github.com/colmap/colmap/blob/main/src/colmap/feature/flexglue.cc
- **配置定义：** https://github.com/colmap/colmap/blob/main/src/colmap/feature/flexglue.h
- **模型来源：** https://github.com/cvg/FlexGlue/releases/tag/v0.1
- **论文：** https://arxiv.org/abs/2610.32976v1
- **核对日期：** 2026-11-08
- **核对范围：** COLMAP 官方 Features 文档的 FlexGlue 节、匹配器配置头文件和 `FlexGlueFeatureMatcher::Match` / `ComputeDenseFeatures` / ONNX 初始化实现。
- **通读：** 已完成
- **映射：** [FlexGlue 统一实体](../../wiki/entities/paper-flexglue.md)

## 已实现的接口

文档列出 `FLEXGLUE` 匹配器类型，适用于任何特征提取器得到的关键点坐标，匹配时忽略已有描述子；图像由 `--FlexGlueMatching.image_path` 提供（必须设置）。需用 `-DONNX_ENABLED=ON` 构建，推荐 GPU。匹配器要求图像文件、非空关键点，guided matching 不支持。

典型调用（需事先运行 feature extraction，使 database.db 含关键点）：

```bash
colmap exhaustive_matcher \
  --database_path "$DATASET_PATH/database.db" \
  --FeatureMatching.type FLEXGLUE \
  --FlexGlueMatching.image_path "$DATASET_PATH/images"
```

COLMAP 的 `FlexGlueMatching.min_covisibility` 默认 **0.5**（0 关闭）；`early_stop_decay` 默认 **0.5**，在 9 层 Transformer 的第 2/4/6/8 层可提前拒绝非共视图像对。注意 **PyTorch FlexGlue() 默认 min_covisibility=0.0**，二者不同。COLMAP 文档另提供 `automatic_reconstructor --feature flexglue` 预设。

## 源码证据与工程边界

- `flexglue.cc` 从 `options_.image_path / image.name` 读取 RGB 图像，经双线性缩放构成 CHW float32 张量，由 ONNX 描述器推理得到稠密特征，并通过 LRU 缓存复用。
- `Match()` 对两个输入关键点集缩放到描述器坐标系（通常 512×512），调用 ONNX matcher，以 `matches0 >= 0` 生成离散点对；共视性、距离与互为最近邻过滤由导出模型负责。
- `flexglue.h` 默认特征缓存为 128 张、每张约 4 MB；CUDA 计算能力 ≥7.0 且 ONNX 可用时优先 fp16，不满足条件回退 fp32。
- 官方文档报告 ETH3D DSLR 22569 对穷举配对，在默认 `min_covisibility=0.5` 与 `early_stop_decay=0.5` 下 **35% 图像对提前停止**；该结果为官方给定配置，不代表任意数据集。
- 已核对的是 COLMAP 当前官方文档与主分支源码，**不等于所有发行版或用户本地预编译包已支持**；未在本环境实际编译 COLMAP 或运行重建。

## 来源链接

- [COLMAP 官方 Feature Extraction and Matching](https://colmap.github.io/features.html#flexglue)
- [COLMAP FlexGlue C++ 匹配器](https://github.com/colmap/colmap/blob/main/src/colmap/feature/flexglue.cc)
- [COLMAP 配置与默认参数](https://github.com/colmap/colmap/blob/main/src/colmap/feature/flexglue.h)
- [FlexGlue 官方 ONNX 推理示例](https://github.com/cvg/FlexGlue/blob/main/onnx/match_pair.py)
