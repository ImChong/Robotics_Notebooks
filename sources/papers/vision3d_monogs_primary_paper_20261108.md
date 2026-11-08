# Gaussian Splatting SLAM

> 来源归档：原作者论文出版入口；针对 3D 视觉与 SLAM 课程目录补齐缺失的主论文，非课程发布者文章。

- **标题：** Gaussian Splatting SLAM
- **类型：** paper
- **原论文：** https://openaccess.thecvf.com/content/CVPR2024/html/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.html
- **官方实现：** https://github.com/VSLAM-LAB/MonoGS
- **会议/期刊：** CVPR 2024
- **入库日期：** 2026-11-08
- **一句话说明：** 单目 Gaussian Splatting SLAM；对当前相机位姿及场景高斯直接优化，结合几何检验与正则化稳定增量重建。
- **核对范围：** 阅读 arXiv 2312.06741v2 方法 §3、实验 Table 1–4、附录 7–12 的实现与局限；对照 https://github.com/VSLAM-LAB/MonoGS README/前后端入口；未独立运行 GPU。
- **通读：** 已完成

## 论文方法和实验核对

- **跟踪**：固定高斯地图，优化当前 SE(3) 相机位姿；单目光度重投影残差，RGB-D 再加入深度残差；论文推导高斯位置与协方差关于相机位姿的解析雅可比。
- **地图**：高斯在当前关键帧共视窗口内增删/优化，通过 covisibility+相对位移选关键帧，几何一致性有 isotropic shape regularization。
- **系统**：单目与 RGB-D 均适用；MonoGS 中 FrontEnd/BackEnd 异步，map 后端长期持有全局高斯集合，不等于多个 Gaussian-SLAM 分块子地图。
- **实验**：论文 Table1 TUM RGB-D ATE RMSE (cm) 仅单目无闭环 fr1/desk 3.78、fr2/xyz 4.60、fr3/office 3.50，平均 3.96；RGB-D 无闭环平均 1.47；Replica RGB-D single-process 平均 0.32cm（Table2），不能互换设置。论文中 RGB-D 单张输出渲染可达较高 FPS，和完整在线 SLAM 速度不同。
- **局限**：论文结论指出目前未集成全局 loop closure，显式 Gaussians 不直接生成显式 surface normals；附录对内存和实时性作独立分析。

## 方法摘要（以出版方摘要为依据）

单目 Gaussian Splatting SLAM；对当前相机位姿及场景高斯直接优化，结合几何检验与正则化稳定增量重建。

## 源码与论文身份

- 原论文：[Gaussian Splatting SLAM](https://openaccess.thecvf.com/content/CVPR2024/html/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.html)
- 代码：[https://github.com/VSLAM-LAB/MonoGS](https://github.com/VSLAM-LAB/MonoGS)
- 总清单：[视觉与三维研究一手来源](../sites/vision3d_research_primary_sources_2026_11_08.md)

## 复现与局限核查

- [x] 核对原论文与附录方法、实验、局限。
- [x] 核对 MonoGS 官方前端/后端关键代码，区别其它 Gaussian-SLAM 同名方法。
- [x] 独立 [MonoGS 深读节点](../../wiki/entities/paper-monogs.md) 包含两类 Mermaid。
