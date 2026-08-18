# DreamX-Phi 1.0: Action-Conditioned Video World Model for Robotic Manipulation（arXiv:2608.13489）

> 来源归档（ingest）

- **标题：** DreamX-Phi 1.0: Action-Conditioned Video World Model for Robotic Manipulation
- **缩写：** **DreamX-Phi**
- **类型：** paper / video-world-model / action-conditioned / manipulation
- **arXiv：** <https://arxiv.org/abs/2608.13489>
- **代码：** <https://github.com/AMAP-ML/DreamX-Phi>（归档见 [`sources/repos/dreamx-phi.md`](../repos/dreamx-phi.md)）
- **作者：** DreamX Team；Rui Chen、Xiangxiang Chu、Geng Li 等
- **机构：** 阿里巴巴（AMAP-ML / 高德）
- **入库日期：** 2026-08-18
- **一句话说明：** 给定观测帧、语言指令与末端位姿+夹爪动作序列，预测未来观测；PRoPE 式 SE(3) 注入保证动作忠实，depth / SAM3 / 冻结 V-JEPA 保几何与小物体，DMD 蒸馏少步部署。

## 开源状态（步骤 2.5）

- **无独立项目页**；以 GitHub 为准。
- **仓库核查（2026-08-18）：** [AMAP-ML/DreamX-Phi](https://github.com/AMAP-ML/DreamX-Phi) MIT，仅 README + LICENSE；写明 **权重与推理代码待 WorldArena 2.0 IROS Challenge 结束后公开**。
- **结论：** **部分开源 / 占位 README**；源码运行时序图不适用。

## 摘录

骨干 Wan2.2-TI2V-5B。真实感 ≠ 动作忠实：错臂、丢物体仍可能「好看」。每臂 SE(3) 经 PRoPE-style 几何编码进 attention。入库时自报 WorldArena 2.0 Track 1 第一、Track 2 并列第二。

**对 wiki 的映射：** [`wiki/entities/paper-dreamx-phi.md`](../../wiki/entities/paper-dreamx-phi.md)；交叉 [Wan](../../wiki/entities/paper-wan-video.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)、[Ctrl-World](../../wiki/entities/paper-ctrl-world.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（占位仓）
