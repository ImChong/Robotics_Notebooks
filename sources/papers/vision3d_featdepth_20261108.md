# FeatDepth — 一手资料归档

> 来源层（原论文或论文出版入口）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** FeatDepth
- **类型：** paper
- **链接：** https://arxiv.org/abs/2007.10603
- **入库日期：** 2026-11-08
- **一句话说明：** 特征度量损失的自监督深度与自运动学习。
- **核对范围：** 已阅读 arXiv 2007.10603v1 全文 16 页，覆盖 §1–3 自监督损失与网络方法、§4.1–4.3 KITTI 深度/里程计 Table 1–4 与消融、§5 结论及参考文献；核对作者官方 README、train.py、cfg_kitti_fm.py、mono/apis/trainer.py、mono/model/mono_fm/net.py 和数据加载入口。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [FeatDepth 原论文](https://arxiv.org/abs/2007.10603) **（本归档主类型）**
- [FeatDepth 源码](https://github.com/sconlyshootery/FeatDepth)

## 论文全文核对要点（arXiv 2007.10603v1，ECCV 2020）

- **核心原因：** 仅用像素光度损失时，低纹理像素的图像梯度接近零，导致几何优化失去有效更新方向（§3.2）。
- **三网结构：** DepthNet 预测单帧深度，PoseNet 估计相邻帧变换，FeatureNet 自编码重建并以特征梯度正则学习更利于优化的特征表示（Fig.1、§3.3–3.4）。
- **损失：** 跨视图 photometric + feature-metric，单视图 autoencoder reconstruction + discriminative + convergent regularization；不同 KITTI 脚本可选择先训练并冻结 FeatureNet（Eq.9、15–18 和 README）。
- **核心指标：** KITTI Table 2 中仅单目训练 M：Abs Rel 0.104、δ1 0.893；双目训练 MS + 在线 refinement：Abs Rel 0.079、δ1 0.925，**两个设置不同不可混为一谈**。
- **作者局限：** Table 3 与 §4.2 提示学习方法的旋转漂移不及传统 ORB-SLAM；在线 refinement 是测试额外计算，复现时需单独标注。

## 方法与边界

特征度量损失的自监督深度与自运动学习。本归档已核对论文方法、KITTI 数字、消融及仓库训练/损失路径；但没有实际训练或评测，运行性能和新环境兼容性仍待验证。

**已核对 README：** 源码仓库自述为 ECCV 2020《Feature-metric Loss for Self-supervised Learning of Depth and Egomotion》官方实现；提供 `train.py`、`config/`、`scripts/infer.py`、`eval_depth.py` 等，且注明模型输出的是 inverse depth。README 列出的老版本 PyTorch/CUDA 是复现兼容性限制。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 已通读 arXiv 2007.10603v1 16 页正文和参考文献。
- [x] 核对官方训练入口、MMCV Runner、数据集和 mono_fm/net.py 的 forward 与两类重投影损失，并绘制来源明确的运行时序图；未执行训练。
- [x] 已升格 [FeatDepth](../../wiki/entities/paper-featdepth.md)，补原文流程图、源码时序图、实验与结论；健康度由 CI/仓库脚本复核。
