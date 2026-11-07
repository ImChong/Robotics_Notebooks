# SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM

> 来源归档：原作者论文出版入口；针对 3D 视觉与 SLAM 课程目录补齐缺失的主论文，非课程发布者文章。

- **标题：** SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM
- **类型：** paper
- **原论文：** https://openaccess.thecvf.com/content/CVPR2024/html/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.html
- **官方实现：** https://github.com/spla-tam/SplaTAM
- **会议/期刊：** CVPR 2024
- **入库日期：** 2026-11-08
- **一句话说明：** 用显式三维高斯进行稠密 RGB-D 追踪与建图；借助轮廓掩码管理场景密度与地图扩展。
- **核对范围：** 已阅读 https://arxiv.org/html/2312.02126 正文、方法、表格、性能分析与附录/局限，核对 https://github.com/spla-tam/SplaTAM 官方源码入口；未独立复现。
- **通读：** 已完成

## 原论文方法与实验事实

在首帧 RGB-D 反投影初始化球形/各向同性 Gaussians；后续每帧先使用 3D Gaussian splatting 可微渲染 RGB、深度和 silhouette，利用 silhouette>阈值的已有地图像素只优化相机位姿，避免未观测区域干扰；再据 silhouette 与深度差把新出现的 RGB-D 表面反投影为新 Gaussians，随后以当前帧和有重叠关键帧优化高斯位置、颜色、尺度、不透明度。每帧 Tracking→Gaussian Densification→Map Updating 是作者 §3 明确顺序。

**实验：** 论文 §5 Table3 ScanNet++ novel-view 合成视角平均 PSNR **24.41 dB**，训练视角平均 **27.98 dB**，两类渲染不应混用；引言/图1 指明 876×584 条件下可达 **400 FPS 的渲染**（不是完整 SLAM Tracking/Mapping 帧率）。论文还在 Replica/TUM RGB-D 上评价 ATE、深度重建。

**局限：** 需要标定过的 RGB-D 相机和相对可靠的深度；大运动、跟踪失败或位姿误差会污染高斯地图；渲染 FPS 不等于完整 SLAM 系统速度。官方代码中基于 torch 和 CUDA renderer 的设置、密度更新与 keyframe 选择必须匹配具体数据集。

## 方法摘要（以出版方摘要为依据）

用显式三维高斯进行稠密 RGB-D 追踪与建图；借助轮廓掩码管理场景密度与地图扩展。

## 源码与论文身份

- 原论文：[SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM](https://openaccess.thecvf.com/content/CVPR2024/html/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.html)
- 代码：[https://github.com/spla-tam/SplaTAM](https://github.com/spla-tam/SplaTAM)
- 总清单：[视觉与三维研究一手来源](../sites/vision3d_research_primary_sources_2026_11_08.md)

## 复现与局限核查

- [x] 已核对论文方法/实验/消融及适用边界。
- [x] 核对原作者仓库主程序与 Mapper/Tracker 实现入口。
- [x] 建立 [SplaTAM](../../wiki/entities/paper-splatam.md) 深读节点。
