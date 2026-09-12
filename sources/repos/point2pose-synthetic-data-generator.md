# hojae-io/Point2Pose-SyntheticDataGenerator

> 来源归档

- **标题：** Point2Pose Synthetic Dataset Generator
- **类型：** repo
- **作者：** Ho Jae Lee（MIT，与 Point2Pose 共同作者）
- **代码：** <https://github.com/hojae-io/Point2Pose-SyntheticDataGenerator>
- **主仓：** <https://github.com/tzuyuan/point-to-pose>
- **论文：** <https://arxiv.org/abs/2604.10415>
- **入库日期：** 2026-09-12
- **一句话说明：** 为 Point2Pose **YCBMultiTrack** 仿真 split 生成 photorealistic RGB-D 与精确位姿/深度的 Isaac Lab 数据管线（与主仓 `experiments/ycbinisaac/` 评测布局对接）。

## 与主项目关系

- 项目页与主 README 将 **YCBMultiTrack** 列为新基准；仿真部分由本仓生成，真机部分为 RealSense + OptiTrack。
- 主仓 `experiments/ycbinisaac/run_ycbinisaac_all.py` 读取 `YCBInIsaacReader` 期望的 per-sequence 目录（`rgb/`、`depth/`、`cam_K.txt`、`masks/`、`annotated_poses/` 等）。

## 开源状态（2026-09-12）

- **已开源：** GitHub 公开仓；与主实现配套，用于复现/扩展仿真评测 split。
- **边界：** 需 Isaac Lab 与 YCB 模型资产；真机 mocap 数据发布路径以主项目页为准。

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-point2pose.md`](../../wiki/entities/paper-point2pose.md) — YCBMultiTrack 数据集节
- 主代码：[`sources/repos/point-to-pose.md`](point-to-pose.md)
- 论文摘录：[`sources/papers/point2pose_arxiv_2604_10415.md`](../papers/point2pose_arxiv_2604_10415.md)
