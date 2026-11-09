# CIMI — Contact Imitation Interface 官方项目主页

- **类型：** site
- **英文标题：** Contact Imitation Interface: Transferring Dexterous Skills by Showing How Contact Unfolds
- **中文标题：** 接触模仿接口：通过接触演化过程迁移灵巧操作技能
- **项目页：** https://nus-lins-lab.github.io/CIMIWeb/
- **论文：** https://arxiv.org/abs/2610.41028
- **补充 PDF：** https://nus-lins-lab.github.io/CIMIWeb/supplement.pdf
- **代码：** https://github.com/ariszxxu/CIMI
- **硬件指南：** 项目页 “Hardware Guide” 链接（Google Docs）
- **入库日期：** 2026-11-03；2026-11-09 复核
- **核对范围：** 通读项目页全部方法、演示、结果与摘要，核对真实 Code 跳转及资源现状；参照同作者论文及官方 RESOURCES.md。
- **通读：** 已完成
- **对 wiki 的映射：** [CIMI 项目实体](../../wiki/entities/paper-contact-imitation-interface.md)

## 项目展示与观察证据

- 页面分为 Demonstration Collection → Contact-Alignment Retargeting + Context-View Reprojection → Policy Training → Inference 四阶段。
- 九路 10 Hz 录制：3 路指根局部接触 RGB，4 路双目场景画面，2 路定位视角。**这不是九路都作为策略输入**；依据任务筛选腕部/虎口与有效手指接触 RGB。
- 视频展示 Coin Picking、1 cm Cube Picking、Letter Retrieval、Bulb Installation（拧入灯泡）以及 LEAP/Wuji 间的人类示教复用。
- 项目页中在六组手—任务对的 20 次/组测试里：接触相机对 ContactSurface 方法使最终阶段平均成功率从 1.7% 到 80.8%；相同相机输入下 ContactSurface vs WristJoint 为 80.8% vs 30.8%。
- 论文 Table III：硬币角点平均误差 14.67→0.51 mm 与灯泡 Light 0→90% 对应**接触优先的角点匹配重定向**，不是相机单独造成。
- 页面的“接触相机”是靠近手指接触区域的 RGB 视觉，**不应表述为直接测得接触力**。

## 真实开源状态

截至 2026-11-09，项目 Code 跳转到 [ariszxxu/CIMI](https://github.com/ariszxxu/CIMI)，仓库实际包含 `src/data_preprocess`、`src/policy`、`src/deploy`、`src/hardware` 等源码，非空仓库。代码、任务配置与安装文档已公开；但 [RESOURCES.md](https://github.com/ariszxxu/CIMI/blob/main/docs/RESOURCES.md) 明确表示原始四套人类示教、六套 Zarr 缓存、论文训练好的权重尚未提供公开下载。训练/真机复现依赖外部数据、硬件与标定。

## 交叉资料

- [原文全文与实验核验](../papers/contact_imitation_interface_arxiv_2610_41028.md)
- [真实源码运行入口](../repos/contact_imitation_interface.md)
- [官方任务矩阵](https://github.com/ariszxxu/CIMI/blob/main/docs/TASKS.md)
- [官方项目页](https://nus-lins-lab.github.io/CIMIWeb/)
