# WiLoR（野外单图 / 视频端到端手部 3D 定位与重建）

- **标题**: WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild
- **论文**: https://arxiv.org/abs/2409.12259 （CVPR 2025）
- **项目页**: https://rolpotamias.github.io/WiLoR/
- **代码**: https://github.com/rolpotamias/WiLoR
- **类型**: paper / code-release / dataset（WHIM 等）
- **作者**: Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, Stefanos Zafeiriou 等
- **收录日期**: 2026-05-07

## 一句话摘要

端到端流水线：**实时全卷积手部检测/定位** + **Transformer 式 3D 手部网格重建**，在野外光照、遮挡与多手下取得较强泛化，并可逐帧用于视频序列（无需显式时序模块亦可形成连贯轨迹）。

## 为何值得保留

- **机器人管线**：单 RGB 下手部 6D/关节或 MANO 类参数是许多遥操作、模仿学习与视频驱动控制分支的输入。
- **数据规模**：作者构建大规模野外手部数据集（论文报告量级达数百万帧级），支撑「in-the-wild」泛化叙事。
- **工程可用**：开源模型与推理优化（如半精度、剪枝）便于嵌入离线视频标注或在线估算。

## 技术要点（来自论文与项目公开描述）

1. **两阶段结构**：检测/定位网络预测手框、左右手类别与 2D keypoints；随后 ViT/MANO 重建网络生成粗手部网格，并用多尺度 vertex-aligned 特征预测残差（检测器自身不输出 3D MANO）。
2. **野外鲁棒**：强调多样光照、遮挡与双手交互场景下的检测稳定性。
3. **效率**：报告在标准基准（如 FreiHAND、HO3D）上的精度与实时性权衡；仓库持续更新推理加速选项。

## 对 Wiki 的映射

- **wiki/methods/wilor.md**：手部 3D 估计方法实体页。
- **wiki/methods/exoactor.md**：作为「生成视频 → 双手姿态 + 语义手势离散化」环节的参照模块。
- **wiki/tasks/manipulation.md**：衔接灵巧操作与视觉感知链路。

## 当前官方推理入口（2026-11-12 静态核验）

- `demo.py`：`ultralytics.YOLO('pretrained_models/detector.pt')` 逐图获得手框/左右手标签 → `wilor.datasets.vitdet_dataset.ViTDetDataset` 裁剪/归一化 → `wilor.models.load_wilor` 载入 `wilor_final.ckpt` → `model(batch)` 输出 mesh/joints/camera → `wilor.utils.renderer.Renderer` 叠加渲染/可选 OBJ 导出。
- `python demo.py --img_folder demo_img --out_folder demo_out --save_mesh --fast` 是 README 可验证的入口；`--fast` 半精度与跳层加速，不是另一套独立模型。
- 权重下载链接、MANO 手模型的额外授权要求与非商用非衍生模型许可须阅读官方 README；不能据开源代码推定商用许可。
- [CVPR 2025 正文及补充材料通读归档](../papers/wilor_cvpr_2025.md)。
