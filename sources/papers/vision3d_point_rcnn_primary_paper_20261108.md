# PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud

> 来源归档：原作者论文出版入口；针对 3D 视觉与 SLAM 课程目录补齐缺失的主论文，非课程发布者文章。

- **标题：** PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud
- **类型：** paper
- **原论文：** https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.html
- **官方实现：** https://github.com/sshaoshuai/PointRCNN
- **会议/期刊：** CVPR 2019
- **入库日期：** 2026-11-08
- **一句话说明：** 第一阶段从原始点云分割前景与背景并生成自下而上的 3D proposals；第二阶段在规范化坐标中提炼候选框并输出精修的框与置信度。
- **核对范围：** 已核对论文出版页面的题名、作者/会议信息及摘要；论文全文、实验表、附录和源码尚未逐一通读。
- **通读：** 未完成

## 方法摘要（以出版方摘要为依据）

第一阶段从原始点云分割前景与背景并生成自下而上的 3D proposals；第二阶段在规范化坐标中提炼候选框并输出精修的框与置信度。

## 源码与论文身份

- 原论文：[PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.html)
- 代码：[https://github.com/sshaoshuai/PointRCNN](https://github.com/sshaoshuai/PointRCNN)
- 总清单：[视觉与三维研究一手来源](../sites/vision3d_research_primary_sources_2026_11_08.md)

## 复现与局限核查

- [ ] 精读正文、实验、附录，核实训练/测试流程、参数与实际性能指标。
- [ ] 核对官方代码真实运行入口，区分论文基线、同仓多个独立模型及第三方复现。
- [ ] 以同一项目的唯一身份更新 wiki 节点；原文驱动 Mermaid 流程图、源码驱动时序图后才可标为 complete。
