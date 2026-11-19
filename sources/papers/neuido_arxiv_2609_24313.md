# NeuIDO：Neural Intrinsic Dynamics Operator for Physics-Informed 4D World Models

> 一手论文来源归档；此页仅提炼研究要点，不转载全文。

- **标题：** NeuIDO: Neural Intrinsic Dynamics Operator for Physics-Informed 4D World Models
- **论文：** <https://arxiv.org/html/2609.24313v1>（arXiv v1，2026-09-21；ECCV 2026）
- **PDF：** <https://arxiv.org/pdf/2609.24313>
- **项目 / 代码：** <https://github.com/JiajingLin/NeuIDO>
- **作者：** Jiajing Lin、Xin Zhang、Jianhua Sun
- **机构：** 上海交通大学人工智能学院、厦门大学信息学院
- **入库日期：** 2026-11-19
- **核对范围：** arXiv HTML v1：§1–§5（相关工作、DeepONet Branch–Trunk、Stage I/II、零样本及 LoRA 适配、Table 1/2 与结果、结论），附录 A–F（数据、经典材料本构、网络架构、其他实验、连续体力学/MPM/3DGS、限制及算子逼近假设）；核对官方 README 与源码
- **通读：** 已完成
- **映射：** [NeuIDO 独立论文详情](../../wiki/entities/paper-neuido-neural-intrinsic-dynamics-operator.md)

## 核心摘录

1. 学习视频函数空间到本构函数空间的算子；Branch 把光流视频映射到材料 latent/系数，弹性与塑性两路 Trunk 在给定形变梯度上输出连续本构响应。
2. Stage I：三类经典材料（弹性、橡皮泥、沙子），各 30 套参数，MPM 共生成 90 条监督轨迹；先学习本构基底，不依赖视频。Stage II：冻结本构基底，利用可微 MPM 与 3DGS 渲染的视觉误差训练视频对齐。
3. 推理：未见场景无需单场景优化的零样本前馈；分布偏移时进一步对 Trunk 加 LoRA 且联合优化 latent，论文设置 100 epoch。
4. Table 1：六个合成场景平均 Chamfer Distance，零样本 1.643、适配后 1.216；NeuMA 1.329、VisionLaw 1.620。论文未证明零样本整体最佳。
5. 泛化：PAC-NeRF 10 个弹性场景 8 训 2 测；Spring-Gaus 真实视频 3 场景、3 视角、每视角 19 帧；已知初速度是附录明确的实验条件。
6. 限制：仅视觉重建监督导致材料物性辨识不唯一，增加视角收益有限；神经本构未硬编码物理守恒，可能把视觉拟合误当正确物性。

## 对 wiki 的映射

- 独立论文唯一实体：[`paper-neuido-neural-intrinsic-dynamics-operator.md`](../../wiki/entities/paper-neuido-neural-intrinsic-dynamics-operator.md)
- [Real2Sim Stage 2](../../roadmap/depth-real2sim.md)：软体本构辨识补齐几何之后的物性，不等于完整 sim-ready 资产。
- [可微仿真](../../wiki/concepts/differentiable-simulation.md)：用 MPM + 可微渲染把视频误差传到可学习的本构关系。
