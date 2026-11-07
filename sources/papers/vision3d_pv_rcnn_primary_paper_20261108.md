# PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection

> 来源归档：原作者论文出版入口；针对 3D 视觉与 SLAM 课程目录补齐缺失的主论文，非课程发布者文章。

- **标题：** PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
- **类型：** paper
- **原论文：** https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.html
- **官方实现：** https://github.com/open-mmlab/OpenPCDet
- **会议/期刊：** CVPR 2020
- **入库日期：** 2026-11-08
- **一句话说明：** 融合稀疏体素特征编码与点集抽象：voxel-to-keypoint 汇聚三维语义，RoI-grid pooling 再完成区域细化。
- **核对范围：** 已核对论文出版页面的题名、作者/会议信息及摘要；论文全文、实验表、附录和源码尚未逐一通读。
- **通读：** 未完成

## 方法摘要（以出版方摘要为依据）

融合稀疏体素特征编码与点集抽象：voxel-to-keypoint 汇聚三维语义，RoI-grid pooling 再完成区域细化。

## 源码与论文身份

- 原论文：[PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.html)
- 代码：[https://github.com/open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- 总清单：[视觉与三维研究一手来源](../sites/vision3d_research_primary_sources_2026_11_08.md)

## 复现与局限核查

- [ ] 精读正文、实验、附录，核实训练/测试流程、参数与实际性能指标。
- [ ] 核对官方代码真实运行入口，区分论文基线、同仓多个独立模型及第三方复现。
- [ ] 以同一项目的唯一身份更新 wiki 节点；原文驱动 Mermaid 流程图、源码驱动时序图后才可标为 complete。
