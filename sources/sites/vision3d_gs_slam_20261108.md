# GS-SLAM — 一手资料归档

> 来源层（作者项目/数据集/官方技术文档）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** GS-SLAM
- **类型：** site
- **链接：** https://gs-slam.github.io/
- **入库日期：** 2026-11-08
- **一句话说明：** 3D Gaussian Splatting 稠密视觉 SLAM。
- **核对范围：** 已读完官网 https://gs-slam.github.io/ 研究简介、方法说明、比较图题、BibTeX 和对外发布物入口，确认源码按钮仅为 Modified Rasterization Code。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [GS-SLAM 项目页](https://gs-slam.github.io/) **（本归档主类型）**
- [GS-SLAM 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_GS-SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting_CVPR_2024_paper.html)

## 作者网站实际发布物

- 官方论文 GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting，CVPR 2024 highlight。
- 项目说明：显式 3D Gaussian、自适应增加/删除噪声高斯、coarse-to-fine tracking。
- **官方源码只有** [Modified Rasterization Code](https://github.com/yanchi-3dv/diff-gaussian-rasterization-for-gsslam)：并无 `run_slam.py`、`tracker.py`、`mapper.py` 这类全系统复现入口，不能虚构源码。
- 官网报告的约 386 FPS 是 **Rendering**，不是完整 SLAM 跟踪+地图更新吞吐。

## 方法与边界

3D Gaussian Splatting 稠密视觉 SLAM。已核对项目官网所有公开研究内容和源码出口；未做运行复现。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 官网正文完整阅读，主论文另见 paper 归档。
- [x] 查明仅修改的 rasterizer 开源，公开代码的调用时序单独核对。
- [x] 已整理至 [GS-SLAM](../../wiki/entities/paper-gs-slam-yan.md)。
