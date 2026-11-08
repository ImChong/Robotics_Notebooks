# SfMLearner — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** SfMLearner
- **类型：** repo
- **链接：** https://github.com/tinghuiz/SfMLearner
- **入库日期：** 2026-11-08
- **一句话说明：** 通过连续视频学习深度与自运动。
- **核对范围：** 完整阅读原作者 README 的准备 KITTI/Cityscapes、训练、深度/位姿评测、模型下载、实际版本说明；核对 train.py、SfMLearner.py::build_train_graph、nets.py::pose_exp_net/disp_net，TensorFlow graph 路径，未运行 TF1.0。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [SfMLearner 源码](https://github.com/tinghuiz/SfMLearner) **（本归档主类型）**
- [SfMLearner 原论文](https://research.google/pubs/unsupervised-learning-of-depth-and-ego-motion-from-video/)

## 作者原始代码入口及复现说明

- `data/prepare_train_data.py` 将 KITTI 原始连续图像变为三帧序列与 K 配置；`train.py` 解析参数后调用 `SfMLearner.train`，TensorFlow 图在 `SfMLearner.py::build_train_graph` 内组合多尺度深度 `disp_net`、位姿/解释性掩码 `pose_exp_net` 和 `projective_inverse_warp`。
- `nets.py` 分别提供 depth decoder 和共享骨干、分叉 pose/mask 预测；`kitti_eval/eval_depth.py` 和 `kitti_eval/eval_pose.py` 分别是指标评测入口。
- 作者 README 说明 **TF1.0/CUDA8/Ubuntu16.04**；新版预训练权重与原论文 Table1 数字可能不同：README KITTI-only 权重 AbsRel 0.183，而论文 v2 KITTI-only 0.208；不能未经版本校验互换。

## 方法与边界

通过连续视频学习深度与自运动。本源码归档已对照作者 README 和主构图入口；论文结果已独立核对原文，但未实际运行 TensorFlow 模型。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 原文通读见配套 [论文归档](../papers/vision3d_sfmlearner_20261108.md)。
- [x] 已核对模型和数据主要入口/调用，见独立节点的源码时序图。
- [x] 已新增 SfMLearner 原始论文节点与 Mermaid 双图。
