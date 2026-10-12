# height-adaptive-planner 仓库归档

- **仓库：** <https://github.com/MingfanZhao/height-adaptive-planner>
- **对应论文：** CMP-IRRT*（[arXiv:2610.10470](https://arxiv.org/abs/2610.10470)）
- **论文归档：** [cmp_irrt_star_arxiv_2610_10470.md](../papers/cmp_irrt_star_arxiv_2610_10470.md)
- **核查日期：** 2026-10-12（`git clone --depth 1`，提交 c71bcd0 “Add files via upload”）
- **许可：** MIT（保留上游 NIRRT*，Zhe Huang 2024，的版权声明）
- **结论：** 部分发布。README 声明提供高度条件碰撞检测、RRT*/IRRT*/CMP 引导 IRRT* 采样逻辑、CMP 网络结构与 NumPy 演示。

## 仓库实际内容

仅 5 个文件：`README.md`、`LICENSE`、`NOTICE.md`、`.gitignore`、`demo_minimal.py`。

- `demo_minimal.py` 从 `height_adaptive_planner` 包导入 `CMPInformedRRTStar2D`、`path_is_collision_free`、`segment_is_free`，用一个手写走廊的 `MockGuide` 做确定性演示（障碍表示 `[x, y, w, h, 估计高度]`，可穿越高度阈值 0.20，低障碍 0.12 可穿越、高障碍 0.80 阻断）。
- **缺失：** README 提到的 `height_adaptive_planner/cmp_model.py` 以及整个 `height_adaptive_planner/` 包在该提交中不存在，因此演示与 CMP 网络均无法直接运行；无训练脚本、无权重、无 Go2 部署代码、无感知（Depth Anything V2 / YOLO）管线。
- README 自述“将继续补充文档、示例和发布资源”；CMP 网络依赖 PyTorch 与 `mamba_ssm`，论文实验环境 Python 3.9.16 + PyTorch 1.13.1 + RTX 4090。

## 映射

- 详情节点：[paper-cmp-irrt-star-height-adaptive-planner.md](../../wiki/entities/paper-cmp-irrt-star-height-adaptive-planner.md)
