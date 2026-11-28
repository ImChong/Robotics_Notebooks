# EgoLoAF：从第一人称示范与手物可供性学习人形移动操作 / Learning Humanoid Loco-Manipulation from Egocentric Demonstrations with Hand-Object Affordances

- **类型：** paper（arXiv 预印本）
- **作者：** Yiming Zeng、Jiangyuxuan Dai、Yuheng Na、Tan Zhu Xuan、Zihan Lin、Xinfang Qi、Shijian Zhang、Chengqi Shi、Yihan Kuang、Zhide Zhong、Junjie He、Junfeng Li、Zimu Han、Naipeng Li、Haoang Li
- **机构：** 西安交通大学、香港科技大学（广州）、Xbotics Research Team
- **论文：** [arXiv:2611.32260v1](https://arxiv.org/abs/2611.32260v1)（2026-11-24） · [原文 PDF](https://arxiv.org/pdf/2611.32260v1)
- **项目页：** <https://egoloaf.github.io/> · [项目页归档](../sites/egoloaf.md)
- **官方代码仓库：** [EgoLoAF-Official](https://github.com/EgoLoAF/EgoLoAF-Official)（截至入库日仅 README）
- **核对范围：** v1 全文 22 页，正文 §1–6，图 1–6、表 1–2，附录 A–E 的采集与坐标约定、可供性编码、部署、分组消融、数据集与质量控制；另核对官方项目页和仓库文件列表。未运行模型、代码或真机。
- **通读：** 已完成
- **入库日期：** 2026-11-28

## 可追溯证据

1. **§3、附录 A–B：** 相机位姿恢复虚拟躯干，HaWoR 重建双手工具中心点，目标使用观测时躯干坐标系，固定至整段结束；30×29 动作块包含三组九维位姿及两个开合状态。当前手物关系是四槽 4×29 条件，不是动作标签；特征和实体掩码分别处理无效值和无效槽。
2. **§5、图 5–6、表 S3–S4：** 四任务平均 TPS 为 EgoLoAF 73.1、无 HOA 遥操作 74.4、有 HOA 遥操作 86.3；同人类数据去 HOA 为 53.2。TPS 是阶段加分，不是成功率。4.03× 是同帧预算下四任务的等权采集人时倍率，未计自动处理或采后审核。
3. **附录 C–E：** 真机以 D435 的实测深度代替离线单目深度；冻结全身控制器以 50 Hz 跟踪，反馈校正仅改变 14 个臂关节目标。论文列 128,400 段 / 1,000 小时 / 1,284 种任务配置；与项目页目前显示待发布的状态区分。
- **对 wiki 的映射：** [EgoLoAF](../../wiki/entities/paper-egoloaf.md)
