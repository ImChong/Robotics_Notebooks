# Distilling Vision-Language Models for Robust Traffic Sign Perception（LAMDA，arXiv:2608.08815）

> 来源归档（ingest）

- **标题：** Distilling Vision-Language Models for Robust Traffic Sign Perception in Autonomous Vehicles
- **缩写 / 框架：** **LAMDA**（Language-Anchored Model for Direction Alignment）
- **类型：** paper / autonomous-driving / vlm-distillation / robustness
- **arXiv：** <https://arxiv.org/abs/2608.08815>
- **会议：** IROS 2026
- **作者：** Pedram MohajerAnsari、Amir Salarpour、Mert D. Pesé
- **机构：** 克莱姆森大学（Clemson）
- **论文宣称代码：** <https://github.com/pedram-mohajer/LAMDA>
- **入库日期：** 2026-08-17
- **一句话说明：** 训练期用冻结 OpenCLIP 文本原型监督交通标志视觉特征，推理丢弃 adapter；不吃对抗样本、不加推理负担。

## 开源状态（步骤 2.5）

- **项目页：** 无独立 github.io。作者主页 [mpese.com 出版物条目](https://mpese.com/publication/mohajeransari-2026-distilling/) 仅会议信息。
- **代码仓核查（2026-08-17）：** 论文写 `github.com/pedram-mohajer/LAMDA`；GitHub API **404**。作者账号 `pedram-mohajer` 存在（ShadowSeq / NS-Attack 等），**无 LAMDA 仓**。
- **结论：** **宣称开源、仓未上线。** 源码运行时序图标 **不适用**。

## 摘录 1：方法

两个固定 prototype bank：VLM 生成的标志描述 + 类别名，经冻结 OpenCLIP 文本编码器。两个辅助损失监督视觉特征（alignment + prototype）。推理只留 backbone + 分类器。

## 摘录 2：数字

GTSRB / LISA，四骨干（ResNet-18/34、Swin-T、ViT-B/16），三种物理可实现攻击（阴影、自然光、RP2 打印补丁）。十种对照里 **唯一** 在全部组合上提升鲁棒性：阴影最高 **+12.5 pp**（GTSRB ResNet-18），自然光最高 **+13.2 pp**（LISA ResNet-34）。干净精度几乎都升或持平（LISA ResNet-18 −0.23 pp）。真机 RP2：37.5%→**75.0%**。最优权重 \(\lambda=\mu=1\)，加大到 2 会伤干净精度。

**对 wiki 的映射：** [`wiki/entities/paper-lamda-tsr.md`](../../wiki/entities/paper-lamda-tsr.md)；交叉 [VLA](../../wiki/methods/vla.md)（语言先验作训练约束，非部署头）。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（GitHub 404）
