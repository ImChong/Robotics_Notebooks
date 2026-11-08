# DUET-DINO: Simultaneous Cross-View World Modeling for Latent Planning in Robot Manipulation（arXiv:2609.10506）

> 来源归档

- **arXiv：** <https://arxiv.org/abs/2609.10506>
- **项目页：** https://utn-air.github.io/DUET-DINO
- **代码：** https://github.com/utn-air/DUET-DINO
- **开源状态：** 待发布
- **入库日期：** 2026-09-10
- **索引来源：** [wechat 11篇盘点](../blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)
- **一句话说明：** 静态侧视+腕部相机联合预测动作条件未来表征，CEM 在 7-DoF 空间做 latent planning；RoboLab reach 92%、angled-reach 72.5%、grasp-and-lift 60%。

**对 wiki 的映射：** [paper-duet-dino](../../wiki/entities/paper-duet-dino.md)

## 2026-11-08 一手资料核对

- **原文：** <https://arxiv.org/html/2609.10506>
- **核对范围：** 完整阅读 v1 I–VI（双视角方法、CEM、仿真、真机和消融；无独立附录），并核验官方项目页和仓库。
- **通读：** 已完成
- **官方代码核验：** <https://github.com/nisarganc/DUET-DINO>；以仓库 README / 可见文件为准，不等同于复现成功。

- **勘误：** 项目页 Code 实际指向 `nisarganc/DUET-DINO`，当前仓库只有 README，标注 `Code coming soon`；非 `utn-air/DUET-DINO`。

## 2026-11-08 论文逐段核验要点

- **方法：** 冻结 DINOv3 ViT-H+/16，对侧视和腕视通过 cross-attention 交互后做 7-DoF 动作条件未来特征预测；CEM 搜索动作序列。
- **实验：** 仿真 Reach 92%、Angled-Reach 72.5%、Lift 60%；真机 Angled-Reach 仅 26.7%（30 次）；仿真每步 CEM 用时约 15–17s。
- **局限：** 作者代码仓 nisarganc/DUET-DINO 目前仅 README 并写 Code coming soon，因此不存在可核验的源码运行调用链。
