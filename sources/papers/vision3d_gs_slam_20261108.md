# GS-SLAM — 一手资料归档

> 来源层（原论文或论文出版入口）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** GS-SLAM
- **类型：** paper
- **链接：** https://openaccess.thecvf.com/content/CVPR2024/html/Yan_GS-SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting_CVPR_2024_paper.html
- **入库日期：** 2026-11-08
- **一句话说明：** 3D Gaussian Splatting 稠密视觉 SLAM。
- **核对范围：** 通读 arXiv:2311.11700v4 HTML 正文 §1–5、Fig 1–6、Table 1–8、附录梯度推导/Algorithm 1、Table 9 和实现细节及参考文献；逐条核对官方项目页和实际只开放的改造栅格化代码 README、Python/CUDA 接口。未运行完整 SLAM。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [GS-SLAM 项目页](https://gs-slam.github.io/)
- [GS-SLAM 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_GS-SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting_CVPR_2024_paper.html) **（本归档主类型）**

## 深读原文证据（Chi Yan et al., CVPR 2024）

1. **§3.1**：显式高斯存 3D center、covariance、opacity 和 SH；RGB 与 depth 均经深度排序 alpha blending 输出（Eq. 1–5）。
2. **§3.2**：初始化时反投影第一帧约 50% RGB-D 像素；每个关键帧据累计不透明度低/渲染深度与传感器深度不符添加新高斯，并按真实深度逐步压低浮空 Gaussians 的 opacity（Eq. 7–9）。
3. **§3.3/补充 Algorithm 1**：用恒速先验给新帧初值，先在 1/2 分辨率粗追踪，再筛选可靠高斯执行 full-res 精追踪；tracking 优化 camera pose **RGB photometric loss**（Eq. 10），BA 先优化 scene、后段联合位姿和场景（Eq. 13）。
4. **Replica 指标**：Table6 8 场景平均 PSNR **34.27dB**、SSIM **0.975**、LPIPS **0.082**、纯渲染 **386.91 FPS**；§4.4 整体 SLAM 约 **8.34 FPS**（RTX 4090/i9-13900K），**不是 386.91 FPS 整套定位建图**。地图约 **198.04 MB**。实验多次重复平均，不代表设备实测。
5. **消融**：删除 adaptive adding 导致场景不全、tracking/mapping 崩溃；删 noise filtering 让 floaters 影响质量；删 coarse-to-fine 降低 ATE/Recall/PSNR（Table7–9）。
6. **源码发布事实**：作者项目页只放了 **Modified Rasterization Code**，不能据此声称完整 GS-SLAM SLAM 框架已开源；官方 rasterizer README 声明 pose 的 depth loss gradient 有问题，实际 tracking 仅 RGB loss。源码时序图因此限于公开 CUDA 渲染/梯度部分。

## 方法与边界

3D Gaussian Splatting 稠密视觉 SLAM。已核对 arXiv v4 论文正文与附录、实验量纲与官方发布代码接口；无法基于未公开的整套 SLAM 源码验证真实 tracker/map 实现。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 已完整阅读 arXiv:2311.11700v4 的方法、实验、算法伪码与附录。
- [x] 核对公开 Python/CUDA rasterizer 核心调用并画时序图；**整套 SLAM 源码未发布，不能补虚假整系统运行时序图**。
- [x] 已完成 [GS-SLAM 作者工作节点](../../wiki/entities/paper-gs-slam-yan.md)，含方法流程图、公开 rasterizer 调用时序、实验与边界。
