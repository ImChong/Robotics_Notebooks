# SfMLearner — Unsupervised Learning of Depth and Ego-Motion from Video

- **作者：** Tinghui Zhou, Matthew Brown, Noah Snavely, David G. Lowe
- **类型：** paper
- **原文：** https://arxiv.org/html/1704.07813 (arXiv v2, CVPR 2017)
- **官方代码：** https://github.com/tinghuiz/SfMLearner
- **入库日期：** 2026-11-08
- **通读：** 已完成
- **核对范围：** 已核读 arXiv v2 的 §1–3.5 网络、loss、梯度问题，§4 KITTI/Make3D/5-frame pose 的 Table1–3 与解释性消融、§5 局限和参考文献；并核对作者官方 README 和训练、模型源码。未运行复现。

## 方法及关键公式

- SfMLearner 用单目视频连续目标帧和来源帧，单目 depth CNN 预测深度、PoseCNN 输出目标到来源的 Euler+translation relative pose；通过可微反投影+双线性采样 inverse warp，将来源图像转为目标视角，用 photometric reconstruction loss 训练（Eq. 1/2）。
- explainability mask 网络学习遮挡/动态像素 soft weighting，并以偏向 mask=1 的交叉熵防止所有 mask=0 的平凡解（Eq. 3）。
- 多尺度图像重建、二阶深度平滑、解释性掩码正则构成总 loss（Eq. 4）；深度与 pose 训练联合，推理可独立。
- 基于已知 K，假设场景大体静态、近似 Lambertian；不直接重建独立运动物体的场景流。

## 实验和指标的正确边界

| 数据集/设置 | 论文结果 |
|---|---:|
| KITTI Eigen 深度（KITTI-only，预测按 GT median scaling） | Abs Rel 0.208、δ1 0.678（Table1） |
| KITTI Eigen 深度（Cityscapes+KITTI） | Abs Rel 0.198、δ1 0.718 |
| Make3D（训练未见，KITTI+Cityscapes，中心裁切/深度小于70m） | AbsRel 0.383（Table2） |
| KITTI odometry 5 帧 ATE：Seq09 / Seq10 | 0.021 / 0.020（Table3，与 ORB-SLAM full 非同样输入） |

README 的另一个 KITTI-only 公开 pretrained checkpoint 报 AbsRel 0.183，是**修改了数据增强和 BN 的较新实现**；不能与论文 v2 表的 0.208 混为完全同设置。论文中比完整序列 ORB-SLAM full ATE 明显更差；mask 网络效果很温和。

## 局限

论文 §5 说明只用静态相机投影假设不能显式估计独立运动/遮挡，且需要已知相机 K。训练域间差距、低纹理、近距离物体是失败例。代码依赖 TF1.0/CUDA8，不代表现有环境直接可复现。

## wiki

[深读独立节点](../../wiki/entities/paper-sfmlearner-zhou.md)
