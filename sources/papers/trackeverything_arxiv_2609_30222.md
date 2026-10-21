# TrackEverything: Long Horizon Dense Tracking via De-Duplicating 3D Scene Representations

> 来源归档（ingest）

- **标题：** TrackEverything: Long Horizon Dense Tracking via De-Duplicating 3D Scene Representations
- **类型：** paper
- **来源：** arXiv:2609.30222（v1，2026-09-24）
- **原始链接：**
  - <https://arxiv.org/abs/2609.30222>
  - HTML：<https://arxiv.org/html/2609.30222v1>
  - 项目页：<https://trackeverything.github.io/>
  - 代码仓：<https://github.com/ayushjain1144/trackeverything>
- **作者：** Ayush Jain, Sreeharsha Paruchuri, Ishita Gupta, Fan Zhang, Tanner Schmidt, Jakob Engel, Katerina Fragkiadaki, Adam W. Harley
- **机构：** 卡内基梅隆大学（Carnegie Mellon University）；Meta
- **入库日期：** 2026-10-16
- **一句话说明：** 将视频逐窗反投影到世界坐标系的持久 3D 场景点云，在窗口边界用体素合并重复表面，并先估计端点/静动属性、再细化动态点轨迹，从而将全可见点长时跟踪的成本更多绑定到独特场景几何。

## 核心论文摘录（MVP）

### 1) 问题：稀疏长时跟踪与稠密短时跟踪的两难

- **链接：** <https://arxiv.org/abs/2609.30222>
- **摘录要点：** 既有稀疏跟踪器可以跨较长视频，但只处理用户指定的少量查询点；稠密跟踪器虽覆盖更多点，却常受限于短片，或只能追踪首帧已出现的内容。TrackEverything 的目标是覆盖视频中所有可见场景点，包括后续才显露的表面。
- **对 wiki 的映射：** [TrackEverything 实体页](../../wiki/entities/paper-trackeverything.md) 说明稀疏/稠密、首帧/全帧设定差别，并与 D4RT、Track4World 对照。

### 2) 方法：世界系点云、端点优先与 3D WAFT

- **链接：** <https://arxiv.org/html/2609.30222v1>
- **摘录要点：** 视频按 16 帧窗口处理，图像特征结合深度与相机位姿（传感器或外部几何估计器，如 VGGT-Ω）反投影为世界坐标系特征点云。端点细化器先预测窗口末端位置、可见性及静态/动态类别；轨迹细化器只为动态点解码窗内路径。3D WAFT 通过在投影位置采样源/目标特征，避免构造昂贵的 4D 相关体。
- **对 wiki 的映射：** 实体页「流程总览」与「核心原理」解释 3D 表示、端点/轨迹解耦和特征采样。

### 3) 去重与长视频效率

- **链接：** <https://arxiv.org/html/2609.30222v1>
- **摘录要点：** 窗口末端把已推进至同一目标时刻的点按预测位置体素化，并对同一体素特征做均值合并；下一窗口只携带压缩后的场景轨迹，并加入新显露内容。作者报告可在 40 GB GPU 显存内跟踪超过 1000 帧视频的所有可见点；在 TAPVid-3D 的短片评测上，相较开放源码的全帧稠密 3D 基线 VDPM 平均 APD-P 高 20% 以上。
- **对 wiki 的映射：** 实体页「实验与结论」记录数字及其评测设定；「局限与风险」说明点云仍随新场景内容增长，且错误合并可能不可逆。

### 4) 复现状态与边界

- **链接：** <https://trackeverything.github.io/>；<https://github.com/ayushjain1144/trackeverything>
- **摘录要点：** 项目页标注 “Code coming soon”，仓库 README 标注代码即将发布；当前仓库仅有 README 与 `assets/`，没有可运行训练、推理或评测入口。因此论文的方法和结果可查，当前尚不能按官方仓库复现。
- **对 wiki 的映射：** 实体页「源码运行时序图」写明不适用，并记录核查日期与仓库状态。

## 关键限制

- 方法依赖外部深度/点图和相机位姿，跟踪性能受输入几何质量影响。
- 窗口间按空间邻近体素合并并做均值池化，若不同表面因定位误差落入同一体素，错误可能被不可逆地带入后续窗口。
- 去重抑制重复观察的冗余，但开放世界探索时活跃场景内容仍持续增长；论文将长期空间缓存淘汰列为后续方向。
- 项目页指出，大位移快速运动、重复纹理或低纹理背景会使 3D 对应变得模糊。

## 关联原始资料

- [TrackEverything 官方项目页](../sites/trackeverything-project.md)
- [TrackEverything 官方代码仓](../repos/trackeverything.md)
