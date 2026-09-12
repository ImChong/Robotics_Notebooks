# point2pose.github.io（Point2Pose 项目页）

- **标题：** Point2Pose: Occlusion-Recovering 6D Pose Tracking and 3D Reconstruction for Multiple Unknown Objects via 2D Point Trackers
- **类型：** site / project-page
- **URL：** <https://point2pose.github.io/>
- **配套论文：** [arXiv:2604.10415](https://arxiv.org/abs/2604.10415) — PDF：<https://arxiv.org/pdf/2604.10415>；归档见 [`sources/papers/point2pose_arxiv_2604_10415.md`](../papers/point2pose_arxiv_2604_10415.md)
- **代码：** <https://github.com/tzuyuan/point-to-pose> — 归档见 [`sources/repos/point-to-pose.md`](../repos/point-to-pose.md)
- **合成数据生成器：** <https://github.com/hojae-io/Point2Pose-SyntheticDataGenerator> — 归档见 [`sources/repos/point2pose-synthetic-data-generator.md`](../repos/point2pose-synthetic-data-generator.md)
- **入库日期：** 2026-09-12

## 一句话摘要

MIT Biomimetic Robotics Lab 与 Boston Dynamics 合作的 **Point2Pose**：从单目 **RGB-D** 视频对 **多个未知刚体** 做 **因果 6D 位姿跟踪** 与 **在线 TSDF 重建**；仅需在物体上点击少量 2D 点初始化，无需 CAD 或类别先验；用 **长程 2D 点跟踪** 维持对应关系，在 **完全遮挡** 后物体重现时可 **即时恢复** 位姿。

## 公开信息要点（截至入库日）

- **机构：** Massachusetts Institute of Technology（Tzu-Yuan Lin、Ho Jae Lee、Yonghyeon Lee、Sangbae Kim）；Boston Dynamics（Kevin Doherty，页注为个人时间完成、与雇主无关）。
- **发表：** **ECCV 2026**（页脚 BibTeX）。
- **初始化：** 每物体 1–3 个图像点（可配合 SAM2 正负 prompt）；无 CAD、无 per-object 训练。
- **管线：** 2D 点跟踪（默认 BootsTAPIR）→ 深度抬升到 3D 关键点图 → 基于图的位姿与地图联合优化 → 在线 per-object TSDF 融合。
- **遮挡恢复：** 点跟踪器在物体重现时重检测对应点，无需单独 relocalization 阶段。
- **多物体：** 手持与机械臂序列中同时跟踪多个物体，经快速运动、遮挡与互遮挡。
- **野外泛化：** 项目页展示香蕉、瓶子、毛绒玩具、手机+AirPods 等日常物体。
- **数据集 YCBMultiTrack：** 仿真（Isaac Lab 精确深度/位姿）+ 真机（RealSense D435i + OptiTrack mocap）；单/双/三物体序列；SAM2 分割 mask、可见性/全遮挡标签。
- **步骤 2.5（开源核查）：** 项目页链到 GitHub [`tzuyuan/point-to-pose`](https://github.com/tzuyuan/point-to-pose)，含 RealSense 实时 demo、数据集评测脚本、模块化 YAML 配置 → **已开源**（BSD 3-Clause）。另链合成数据生成器仓；2026-08 更新 **model-based** 变体（Gaussian splat / 给定 mesh 跟踪）。

## 为何值得保留

- **操作感知接口：** 把「未知物体 6D 轨迹 + 在线 mesh」从 FoundationPose / CAD 依赖路线中解耦，适合 clutter 操作与多物体场景理解。
- **模块化工程：** segmenter / tracker / register / optimizer / reconstructor 均可 YAML 互换，便于与机器人感知栈对照选型。
- **评测基准：** YCBMultiTrack 填补多物体动态 RGB-D + mocap 空白（相对 HO3D 单物体、YCB-Video 静态、HOT3D 缺稠密深度）。

## 关联资料

- 论文归档：[`sources/papers/point2pose_arxiv_2604_10415.md`](../papers/point2pose_arxiv_2604_10415.md)
- 主代码仓：[`sources/repos/point-to-pose.md`](../repos/point-to-pose.md)
- 合成数据：[`sources/repos/point2pose-synthetic-data-generator.md`](../repos/point2pose-synthetic-data-generator.md)
- 升格：[`wiki/entities/paper-point2pose.md`](../../wiki/entities/paper-point2pose.md)
