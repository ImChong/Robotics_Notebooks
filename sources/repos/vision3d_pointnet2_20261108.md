# PointNet++ — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** PointNet++
- **类型：** repo
- **链接：** https://github.com/charlesq34/pointnet2
- **入库日期：** 2026-11-08
- **一句话说明：** 分层点集特征抽象与局部几何学习。
- **核对范围：** 官方 README 全文与真实训练、模型与评测入口已核对：train.py → models/pointnet2_cls_ssg.py 或 pointnet2_cls_msg.py get_model → pointnet_sa_module_msg / pointnet_sa_module → get_loss → TensorFlow optimizer.minimize；evaluate.py 单独读测试集。；读数来自原论文而非复现。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [PointNet++ 源码](https://github.com/charlesq34/pointnet2) **（本归档主类型）**

## README 与代码核对

train.py → models/pointnet2_cls_ssg.py 或 pointnet2_cls_msg.py get_model → pointnet_sa_module_msg / pointnet_sa_module → get_loss → TensorFlow optimizer.minimize；evaluate.py 单独读测试集。

**方法**：每个 Set Abstraction（SA）层通过 Farthest Point Sampling（FPS）选中心点，ball query 在固定空间半径内找邻居；邻域点转为相对中心的坐标，局部 PointNet MLP+max-pool 聚合，再递归 SA 扩大感受野。Multi-Scale Grouping（MSG）用多个半径编码同一中心点并随机丢点训练，Multi-Resolution Grouping（MRG）融合跨层尺度以降低计算成本。分割时 Feature Propagation 将粗层表征插值回原始点。

**限制**：MSG 多半径局部特征计算较重，ball query 半径需要依据实际采样密度调整。旧 TF1.2 GPU/Python2.7 环境不保证在现有 GPU 驱动直接运行；表中含 normal 的结果不等同于 xyz-only 输入。

## 方法与边界

分层点集特征抽象与局部几何学习。已核对官方 README 与主要执行模块，并依据原文核对研究方法，未在本次运行指标复现。

**已核对 README：** 官方 README 指向 NIPS 2017 原论文，提供 `train.py`、`train_multi_gpu.py`、`evaluate.py` 等实验入口；依赖旧版 TensorFlow，需要独立评估运行环境。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 已通读对应原论文并建立独立归档。
- [x] 已核对源码入口和模型路径，时序图见 wiki。
- [x] 已建立 [PointNet++ 深读节点](../../wiki/entities/paper-pointnet2.md)，健康度待 CI 复核。
