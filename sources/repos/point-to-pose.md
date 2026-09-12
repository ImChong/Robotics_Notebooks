# tzuyuan/point-to-pose

> 来源归档

- **标题：** Point2Pose 官方实现
- **类型：** repo
- **组织 / 作者：** Tzu-Yuan Lin、Ho Jae Lee、Yonghyeon Lee、Sangbae Kim（MIT）；Kevin Doherty（Boston Dynamics，README 注个人时间）
- **代码：** <https://github.com/tzuyuan/point-to-pose>
- **项目页：** <https://point2pose.github.io/>
- **论文：** <https://arxiv.org/abs/2604.10415>
- **许可：** BSD 3-Clause（`LICENSE`）；部分第三方权重 CC BY-NC（CoTracker3 / LiteTracker）
- **入库日期：** 2026-09-12
- **一句话说明：** 无 CAD 的多物体 RGB-D **6D 位姿跟踪** + **在线 TSDF 重建**：模块化 YAML 管线（SAM2 分割、可换点跟踪器、图优化、SDF 融合）；含 RealSense 实时 demo、HO3D / YCBInEOAT / YCBMultiTrack 评测脚本。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `conda env create -f environment.yml` | Ubuntu 22.04 · Python 3.11 · PyTorch 2.4 · CUDA 12.1（论文环境） |
| `pip install --no-build-isolation -r requirements-third-party.txt` | SAM2-realtime、tapnet（BootsTAPIR）、LightGlue |
| `checkpoints/` | SAM2、TAPIR、可选 TAPNext / Track-On2 权重 |
| `examples/realsense_tracking/realsense_tracking.py` | RealSense 2D 实时跟踪 UI |
| `examples/realsense_tracking/realsense_tracking_3d.py` | 同上 + Rerun 3D 可视化 |
| `examples/realsense_tracking/record_rgbd.py` | 录制 YCBMultiTrack 布局 RGB-D |
| `experiments/ho3d/run_ho3d_single.py` | HO3D-v3 单序列评测 |
| `experiments/ycbineoat/run_ycbineoat_all.py` | YCBInEOAT 评测 |
| `experiments/ycbinisaac/run_ycbinisaac_all.py` | YCBMultiTrack 仿真+真机评测 |
| `configs/pipeline/pipeline_test2.yaml` | 带注释的参考管线配置 |
| `configs/ho3d_exp/eccv_final.yaml` 等 | 论文复现设置 |
| `point2pose/pipeline/` | `ModularPipeline` 组装 segmenter / tracker / register / optimizer / reconstructor |
| `examples/model_based_tracking/` | 2026-08 **model-based** 变体（给定 mesh / Gaussian） |
| `examples/realsense_tracking/reconstruction/` | 3D Gaussian splat 重建管线 |

## 可互换模块（registry）

| 模块 | 默认 / 常用 `type` | 备注 |
|------|-------------------|------|
| `segmenter` | `sam2` | 实时 SAM2 fork |
| `tracker` | `tapir`（BootsTAPIR） | 论文主结果；可选 `tapnext`、`trackon`、`litetracker`、`cotracker3_online` |
| `sampler` | `super_point_balanced` 等 | SuperPoint / FPS 关键点采样 |
| `register` | `svd_residual_outlier` 等 | 3D 关键点图配准 |
| `local_optimizer` / `global_optimizer` | `lm_graph` / `isam2` | 图优化位姿与地图 |
| `reconstructor` | `sdf_builder` | 在线 TSDF 融合 |

## 硬件与依赖边界

- **GPU：** NVIDIA CUDA（RealSense demo ≥8 GB 显存推荐；论文 RTX 4090）。
- **相机：** Intel RealSense D435i / D455（`pyrealsense2`）。
- **配置路径：**  shipped YAML 含作者机器绝对路径，运行前需改 `checkpoint_path`、`debug_dir`、`pose_save_path`。
- **已知问题：** torchvision 先于 `cv2.namedWindow` 导入可能挂死；SAM2 模块 import 时启用全局 bf16 autocast。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Point2Pose](../../wiki/entities/paper-point2pose.md) | 论文实体：方法、数据集、局限 |
| [Grasp Pose Estimation](../../wiki/methods/grasp-pose-estimation.md) | 同为 6-DoF 感知，但 Point2Pose 侧重 **跟踪+重建** 而非抓取候选 |
| [Embodied Perception Six Spatial Representations](../../wiki/concepts/embodied-perception-six-spatial-representations.md) | RGB-D / TSDF / 物体坐标系表示选型 |
| [Manipulation](../../wiki/tasks/manipulation.md) | 多物体 clutter 操作的上游位姿感知 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/point2pose_arxiv_2604_10415.md`](../papers/point2pose_arxiv_2604_10415.md)
- 项目页：[`sources/sites/point2pose.md`](../sites/point2pose.md)
- 合成数据：[`sources/repos/point2pose-synthetic-data-generator.md`](point2pose-synthetic-data-generator.md)
- 沉淀 **[`wiki/entities/paper-point2pose.md`](../../wiki/entities/paper-point2pose.md)**
