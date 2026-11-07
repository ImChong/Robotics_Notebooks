# Gaussian Splatting SLAM

> 来源归档：原作者论文出版入口；针对 3D 视觉与 SLAM 课程目录补齐缺失的主论文，非课程发布者文章。

- **标题：** Gaussian Splatting SLAM
- **类型：** paper
- **原论文：** https://openaccess.thecvf.com/content/CVPR2024/html/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.html
- **官方实现：** https://github.com/VSLAM-LAB/MonoGS
- **会议/期刊：** CVPR 2024
- **入库日期：** 2026-11-08
- **一句话说明：** 单目 Gaussian Splatting SLAM；对当前相机位姿及场景高斯直接优化，结合几何检验与正则化稳定增量重建。
- **核对范围：** 已核对论文出版页面的题名、作者/会议信息及摘要；论文全文、实验表、附录和源码尚未逐一通读。
- **通读：** 未完成

## 方法摘要（以出版方摘要为依据）

单目 Gaussian Splatting SLAM；对当前相机位姿及场景高斯直接优化，结合几何检验与正则化稳定增量重建。

## 源码与论文身份

- 原论文：[Gaussian Splatting SLAM](https://openaccess.thecvf.com/content/CVPR2024/html/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.html)
- 代码：[https://github.com/VSLAM-LAB/MonoGS](https://github.com/VSLAM-LAB/MonoGS)
- 总清单：[视觉与三维研究一手来源](../sites/vision3d_research_primary_sources_2026_11_08.md)

## 复现与局限核查

- [ ] 精读正文、实验、附录，核实训练/测试流程、参数与实际性能指标。
- [ ] 核对官方代码真实运行入口，区分论文基线、同仓多个独立模型及第三方复现。
- [ ] 以同一项目的唯一身份更新 wiki 节点；原文驱动 Mermaid 流程图、源码驱动时序图后才可标为 complete。
