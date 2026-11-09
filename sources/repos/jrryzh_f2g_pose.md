# jrryzh/F2G-Pose — 官方源码与部署入口

- **类型：** repo
- **仓库：** https://github.com/jrryzh/F2G-Pose
- **论文：** https://arxiv.org/abs/2610.31623
- **项目：** https://jrryzh.github.io/F2G-Pose/
- **模型：** https://huggingface.co/J3rr1/F2G-Pose
- **入库日期：** 2026-11-09
- **许可：** F2G-Pose 集成代码 Apache-2.0；third_party/RADIO/Omni6DPose 等单独遵守原许可
- **核对范围：** 阅读官方 README、pyproject.toml、docs/INSTALL.md、USAGE.md、EVALUATION.md、VALIDATION.md；检查 src/f2g_pose/cli.py、estimator.py、preprocess.py 和 model.py 的真实推理/输出代码路径。这是推理路径复核，不宣称所有文件逐行审计、实机运行或重新复现 benchmark。
- **通读：** 已完成

## 运行入口与源码调用

| 入口 | 对应文件与行为 |
|---|---|
| `f2g-pose predict` | `pyproject.toml` script → `f2g_pose.cli:main`，`io.load_inputs` 读取 RGB、depth、mask、K |
| `PoseEstimator.predict` | `estimator.py` 调用 `preprocess.prepare`，然后 `forward` |
| `preprocess.prepare` | 验证 H×W 对齐、相机内参、depth_scale；mask 有至少 50 个有效深度点；裁切到 336、采样 1024 点、减去观测中心 |
| `RGBDEncoder.forward` | `model.py`：冻结 RADIO 第 8/16/23 层 → 跨模态特征融合 → DGCNN token → MoE Transformer → 全局物体特征 |
| `F2GPose.forward` | `model.py`：6D 旋转头、平移头、尺寸头；可选 2048 点形状头 |
| `PoseEstimator.decode` | 6D 旋转正交化，平移与形状加回观测中心 → `T_cam_obj` / `size_m` / `points_cam_m` |
| `save_prediction` | 预测 JSON + 相机坐标 PLY；`--pose-only` 关闭 shape |
| `f2g-pose demo` | `demo.launch`，可上传对齐 RGB-D / mask 或选用额外 SAM checkpoint 生成 mask |
| `f2g-pose train/eval/benchmark` | `cli.py` 分别分派 `train.train` / `evaluate.evaluate` / `benchmark.benchmark` |

## 复现指令（官方）

```bash
# Linux、NVIDIA GPU，Python 3.10、PyTorch 2.4.0、CUDA 12.4；需安装 CUDA 扩展及单独下载 RADIO / F2G-Pose 权重
f2g-pose demo --checkpoint weights/f2g-pose.pt --port 7860
f2g-pose predict --checkpoint weights/f2g-pose.pt \
  --rgb example/rgb.png --depth example/depth.npy --mask example/mask.png \
  --intrinsics example/K.txt --depth-scale 1 --output outputs/example
```

真实样例在 `examples/rope/`；`depth_scale=0.001` 对应毫米原始深度，`1.0` 对应米。相机坐标轴 X 向右、Y 向下、Z 向前；输出物体到相机的 4×4 变换。

## 论文 / release 指标版本区分

| 来源 | ROPE IoU50 AUC | SOPE IoU50 AUC | RTX 4090 模型 FPS |
|---|---:|---:|---|
| arXiv v1 Table 1/4 | 27.8 | 43.7 | full 23.7 / pose-only 28 |
| 公开权重 full-test（README、EVALUATION.md）| 28.19 | 44.73 | 不在该 eval protocol 报告 |
| 当前 README batch-1 FP32 同步计时 | — | — | full 36.6 / pose-only 37.2 |

源码的 `docs/EVALUATION.md` 说明完整测试集选用 BF16、H200、batch 128 与完整覆盖；README 的新 FPS 排除 SAM 和前处理。不同口径不能直接给出同条件优越结论。

## 边界

- 预处理强依赖对齐深度、目标 mask、内参及单位；少于 50 个有效前景深度像素会报错。
- GPU/CUDA 扩展是硬性前提；预训练 RADIO 和 F2G-Pose checkpoints 需单独准备。
- 官方 VALIDATION 声明通过 15 个测试、短程训练、样例预测；Docker 实际执行和托管 ZeroGPU 在该文档中明确为未测。此归档未本地运行代码。

## 对 wiki 的映射

- [F2G-Pose 实体](../../wiki/entities/paper-f2g-pose.md)
- [论文](../papers/f2g_pose_arxiv_2610_31623.md)
- [项目主页](../sites/f2g-pose.md)
- [抓取位姿估计](../../wiki/methods/grasp-pose-estimation.md)
