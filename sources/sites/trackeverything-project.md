# TrackEverything — 官方项目页

- **来源：** <https://trackeverything.github.io/>
- **类型：** site（论文项目页）
- **机构：** 卡内基梅隆大学（Carnegie Mellon University）；Meta
- **作者：** Ayush Jain, Sreeharsha Paruchuri, Ishita Gupta, Fan Zhang, Tanner Schmidt, Jakob Engel, Katerina Fragkiadaki, Adam W. Harley
- **归档日期：** 2026-10-16
- **论文：** arXiv:2609.30222 — *TrackEverything: Long Horizon Dense Tracking via De-Duplicating 3D Scene Representations*
- **代码：** <https://github.com/ayushjain1144/trackeverything>
- **开源结论（步骤 2.5）：** **宣称将开源 / 待发布** — 项目页写明 “Code coming soon”；截至 2026-10-16，仓库仅有 README 与 `assets/`，无可运行实现。

## 一句话说明

TrackEverything 以世界坐标系中的持久 3D 场景轨迹做视频全可见点跟踪，项目页展示超过 1000 帧长视频、静态/动态分类及与 DELTA-v2、CoTracker3、SpatialTracker-v2 的定性对比。

## 页面信息摘录

- 方法从新内容出现时开始建立轨迹，不限于首帧可见点；展示图中为清晰起见只画动态点轨迹，静态点仍被追踪并计入场景点云。
- 项目页列出失败情形：快速大位移，以及重复或低纹理背景下的 3D 对应歧义。
- 页面提供长视频、定性结果、静态/动态分类、基线对比及 BibTeX。
- 页面没有训练、推理、数据或权重下载链接；代码发布状态以页面与官方仓库后续更新为准。

## 对 wiki 的映射

- [TrackEverything 论文实体页](../../wiki/entities/paper-trackeverything.md) — 统一编译论文、项目页和源码状态。
- [状态估计](../../wiki/concepts/state-estimation.md) — 将其纳入动态 3D 几何与点跟踪谱系。

## 关联原始资料

- [TrackEverything 论文摘录](../papers/trackeverything_arxiv_2609_30222.md)
- [TrackEverything 代码仓](../repos/trackeverything.md)
