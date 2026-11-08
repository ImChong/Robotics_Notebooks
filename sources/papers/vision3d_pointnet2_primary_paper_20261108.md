# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

> 来源归档：原作者论文出版入口；针对 3D 视觉与 SLAM 课程目录补齐缺失的主论文，非课程发布者文章。

- **标题：** PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
- **类型：** paper
- **原论文：** https://arxiv.org/abs/1706.02413
- **官方实现：** https://github.com/charlesq34/pointnet2
- **会议/期刊：** NeurIPS 2017 / arXiv 1706.02413
- **入库日期：** 2026-11-08
- **一句话说明：** 以递归局部点集抽象组织层级表示，并针对非均匀点密度使用多尺度特征聚合。
- **核对范围：** 已按 https://arxiv.org/html/1706.02413 阅读原论文全文方法、实验、消融/参考与公开附录，并核对作者 README 和真实源码入口；未运行程序。
- **通读：** 已完成

## 原始论文方法与结果核对

每个 Set Abstraction（SA）层通过 Farthest Point Sampling（FPS）选中心点，ball query 在固定空间半径内找邻居；邻域点转为相对中心的坐标，局部 PointNet MLP+max-pool 聚合，再递归 SA 扩大感受野。Multi-Scale Grouping（MSG）用多个半径编码同一中心点并随机丢点训练，Multi-Resolution Grouping（MRG）融合跨层尺度以降低计算成本。分割时 Feature Propagation 将粗层表征插值回原始点。

**实验/消融：** arXiv v1 Table2 ModelNet40 点坐标输入：PointNet 89.2%、PointNet++ 90.7% 分类准确率；输入增加法线并用 5000 点时达到 91.9%，不能与前者视为同条件。论文另在 ScanNet、SHREC15 和附录分割、kNN/ball query、FPS 随机性进行分析。

**源码：** train.py → models/pointnet2_cls_ssg.py 或 pointnet2_cls_msg.py get_model → pointnet_sa_module_msg / pointnet_sa_module → get_loss → TensorFlow optimizer.minimize；evaluate.py 单独读测试集。

**局限：** MSG 多半径局部特征计算较重，ball query 半径需要依据实际采样密度调整。旧 TF1.2 GPU/Python2.7 环境不保证在现有 GPU 驱动直接运行；表中含 normal 的结果不等同于 xyz-only 输入。

## 方法摘要（以出版方摘要为依据）

以递归局部点集抽象组织层级表示，并针对非均匀点密度使用多尺度特征聚合。

## 源码与论文身份

- 原论文：[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
- 代码：[https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- 总清单：[视觉与三维研究一手来源](../sites/vision3d_research_primary_sources_2026_11_08.md)

## 复现与局限核查

- [x] 已读原文方法、实验/消融、附录及限制。
- [x] 已核对官方源码真实训练/网络入口，不混同共享仓库内不同模型。
- [x] 生成独立 [PointNet++](../../wiki/entities/paper-pointnet2.md)，含流程图与源码时序图。
