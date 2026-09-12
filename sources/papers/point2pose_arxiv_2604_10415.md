# Point2Pose: Occlusion-Recovering 6D Pose Tracking and 3D Reconstruction for Multiple Unknown Objects via 2D Point Trackers（arXiv:2604.10415）

> 来源归档（ingest）

- **标题：** Point2Pose: Occlusion-Recovering 6D Pose Tracking and 3D Reconstruction for Multiple Unknown Objects via 2D Point Trackers
- **类型：** paper / 6D pose tracking / RGB-D / multi-object / reconstruction
- **arXiv：** <https://arxiv.org/abs/2604.10415>（PDF：<https://arxiv.org/pdf/2604.10415>）
- **项目页：** <https://point2pose.github.io/>
- **代码：** <https://github.com/tzuyuan/point-to-pose>
- **合成数据生成器：** <https://github.com/hojae-io/Point2Pose-SyntheticDataGenerator>
- **作者：** Tzu-Yuan Lin；Ho Jae Lee；Kevin Doherty；Yonghyeon Lee；Sangbae Kim
- **机构：** Massachusetts Institute of Technology；Boston Dynamics（Doherty）
- **发表：** European Conference on Computer Vision（**ECCV 2026**）
- **入库日期：** 2026-09-12
- **一句话说明：** **无模型** 因果 **多物体 6D 位姿跟踪**：稀疏 2D 点初始化 + 长程 **2D 点跟踪器** 维持对应 → 深度抬升 3D 关键点图 + 图优化 → 同时 **在线 TSDF** 重建；**完全遮挡** 后即时恢复；发布 **YCBMultiTrack** 多物体 RGB-D 基准（仿真 + mocap 真机）。

## 开源状态（项目页 + 仓库核查，2026-09-12）

- **已开源、可运行：** 项目页与 README 链到 [`tzuyuan/point-to-pose`](https://github.com/tzuyuan/point-to-pose)。`conda env create -f environment.yml`；RealSense 实时 demo（`examples/realsense_tracking/`）；HO3D / YCBInEOAT / YCBMultiTrack 评测脚本；模块化 YAML 管线。
- **部分依赖需自建：** SAM2-realtime、tapnet、LightGlue 从源码安装；checkpoint 需手动下载；shipped 配置含绝对路径。
- **扩展（2026-08）：** model-based 跟踪（给定 mesh / Gaussian）与 3DGS 重建子目录。
- **合成数据：** [`hojae-io/Point2Pose-SyntheticDataGenerator`](https://github.com/hojae-io/Point2Pose-SyntheticDataGenerator) 公开。
- **许可：** BSD 3-Clause（主仓）；CoTracker3 / LiteTracker 权重 **CC BY-NC**。

## 摘要级要点

- **问题：** 现有无 CAD 跟踪器多依赖帧间匹配，**完全遮挡** 或离视野后难以恢复；单物体精度与多物体/遮挡鲁棒性难以兼得。
- **Point2Pose 思路：** 用学习式 **长程 2D 点跟踪**（默认 BootsTAPIR）作持久数据 association；2D 轨迹 + RGB-D 深度构建 per-object **关键点图**；**图优化** 联合 refine 位姿与地图；深度融合 **在线 TSDF**。
- **初始化：** 用户点击少量图像点（配合 SAM2 分割）；**无需** 物体 CAD、类别先验或 per-object 训练。
- **遮挡：** 点跟踪器在物体重现时重检测对应点 → **即时** 恢复 6D 位姿，无单独 relocalization。
- **权衡：** 实验显示相对部分单物体 SOTA **牺牲少许单物体精度**，换取多物体跟踪与完全遮挡恢复能力。
- **基准 YCBMultiTrack：** YCB 子集；仿真（Isaac Lab）+ 真机（RealSense D435i + OptiTrack）；单/双/三物体独立运动；互遮挡与全视野遮挡；SAM2 mask、可见性标签。

## 核心论文摘录（MVP）

### 1) 长程 2D 点跟踪 → 3D 关键点图与 6D 配准

- **链接：** 项目页 “How It Works”；论文方法节
- **摘录要点：** 学习式点跟踪器维护跨数百帧的像素对应；深度将跟踪点抬升为 3D，形成 per-object keypoint map；通过 map-based registration 恢复每物体 6-DoF 位姿。
- **对 wiki 的映射：**
  - [Point2Pose](../../wiki/entities/paper-point2pose.md) — 流程总览与源码时序图。
  - [Embodied Perception Six Spatial Representations](../../wiki/concepts/embodied-perception-six-spatial-representations.md) — RGB-D 物体坐标表示。

### 2) 图优化 + 在线 TSDF 重建

- **链接：** 项目页 pipeline 图；README Configuration & Architecture
- **摘录要点：** 位姿与关键点图联合图优化（`lm_graph` / `isam2` 等可换）；估计位姿同时将深度融合为 per-object **Truncated SDF**，边跟踪边生成纹理 mesh。
- **对 wiki 的映射：**
  - [Point2Pose](../../wiki/entities/paper-point2pose.md) — 工程实践与模块 registry 表。
  - [Grasp Pose Estimation](../../wiki/methods/grasp-pose-estimation.md) — 6-DoF 感知任务对照（抓取 vs 跟踪）。

### 3) 完全遮挡恢复与 YCBMultiTrack 基准

- **链接：** 项目页 “Instant Recovery”；“YCBMultiTrack Dataset”
- **摘录要点：** 传统帧间匹配在完全遮挡后失效；Point2Pose 依赖点查询重检测实现 **instant** 恢复。YCBMultiTrack 相对 HO3D（单物体）、YCB-Video（静态）、HOT3D（缺稠密深度）强调 **多物体动态 + 遮挡 + mocap GT**。
- **对 wiki 的映射：**
  - [Point2Pose](../../wiki/entities/paper-point2pose.md) — 局限（单物体精度权衡）与结论。
  - [Manipulation](../../wiki/tasks/manipulation.md) — clutter 多物体操作场景。

## BibTeX

```bibtex
@inproceedings{lin2026point2pose,
  title     = {Point2Pose: Occlusion-Recovering 6D Pose Tracking and 3D Reconstruction
               for Multiple Unknown Objects via 2D Point Trackers},
  author    = {Lin, Tzu-Yuan and Lee, Ho Jae and Doherty, Kevin and Lee, Yonghyeon and Kim, Sangbae},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-point2pose.md`](../../wiki/entities/paper-point2pose.md)
- 代码归档：[`sources/repos/point-to-pose.md`](../repos/point-to-pose.md)
- 合成数据：[`sources/repos/point2pose-synthetic-data-generator.md`](../repos/point2pose-synthetic-data-generator.md)
- 项目页：[`sources/sites/point2pose.md`](../sites/point2pose.md)
- 互链：[Grasp Pose Estimation](../../wiki/methods/grasp-pose-estimation.md)、[Embodied Perception Six Spatial Representations](../../wiki/concepts/embodied-perception-six-spatial-representations.md)、[Manipulation](../../wiki/tasks/manipulation.md)
