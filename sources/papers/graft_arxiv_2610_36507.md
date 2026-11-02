# GRAFT: Geometry-Aware Residual Adaptation for Foundation Trackers

> 来源归档（paper）

- **论文：** <https://arxiv.org/abs/2610.36507v1>
- **全文：** <https://arxiv.org/html/2610.36507v1>
- **作者：** Liang Li, Guoyao Zhang, Xingyu Chen, Sikai Wu, Qianyang Wu, Yuhang Lin, Renjing Xu, Mingliang Zhou, Diyun Xiang
- **单位：** 小米机器人实验室、香港科技大学（广州）
- **版本：** arXiv v1，2026-10-27
- **项目页：** <https://graft-humanoid.github.io/>
- **入库日期：** 2026-11-02
- **一句话：** 冻结行为跟踪器，利用地形条件下肢残差适配复杂地形。

## 原文核查

通读 v1 方法、实验和结论：特权高程教师以 PPO 训练，深度学生模仿教师；每个基础跟踪器单独训练适配器。实验包括 Isaac Lab 训练、MuJoCo 评测和 G1 真机展示。项目页源码仅展示论文与演示入口，未发现 GRAFT 算法实现或权重入口，开放状态尚不能断言。

## 对 wiki 的映射

- [GRAFT 项目实体](../../wiki/entities/paper-graft.md)
