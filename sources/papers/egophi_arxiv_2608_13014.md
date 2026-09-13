# EgoPHI: Estimating 3D Hand-Object Contact and Force from Egocentric Vision

> 来源归档

- **标题：** EgoPHI: Estimating 3D Hand-Object Contact and Force from Egocentric Vision
- **类型：** paper / egocentric vision / hand-object interaction / contact / force estimation
- **arXiv：** <https://arxiv.org/abs/2608.13014>（2026）
- **PDF：** <https://arxiv.org/pdf/2608.13014>
- **项目页：** <https://siplab.org/projects/EgoPHI>
- **代码：** <https://github.com/eth-siplab/EgoPHI>
- **数据 / 力标注：** <https://huggingface.co/datasets/eth-siplab/EgoPHI>
- **作者：** Andela Ilic, Rachel Schuchert, Yijing Jiang, Christian Holz
- **机构：** 苏黎世联邦理工（ETH Zürich）；Sensing, Interaction & Perception Lab（SIPLAB）
- **出处：** ECCV 2026
- **入库日期：** 2026-09-13
- **一句话说明：** 首个从单目 ego RGB + 物体几何联合估计双手–关节物体 mesh 上稠密 3D 接触与力分布的视觉模型；SOFA 物理仿真生成力监督，ARCTIC 训练、H2O 跨数据集与自研真机物体验证 sim-to-real。

---

## 摘要级要点

1. **问题：** ego 手–物交互理解需超越「接触在哪」到「施力多少、如何施力」；同类抓取可接触相近但力分布不同。
2. **方法：** 三阶段 **InteractionGNN** 管线——(1) 视觉/几何特征与跨模态融合；(2) **IRM** 迭代细化物体位姿（应对手部遮挡）；(3) **Graph-Based Interaction Blocks** 在双手+物体 mesh 上预测 per-vertex 接触与 3D 力。
3. **监督：** 现有 HOI 数据集缺可扩展力真值 → 提出 **SOFA 物理仿真管线**，为 ARCTIC 等数据生成 per-vertex 力标注（HF 发布预计算力场）。
4. **训练/评测：** 仅在 **ARCTIC** 训练；**ARCTIC s05** 域内评测 + **H2O s4_ego** 跨数据集泛化；对比 **PressureVision**（2D 力）与 **HACO**（3D mesh 级）。
5. **主要结果（项目页）：** ARCTIC 上手部力 **MAE 4.03 N**（HACO 6.62 N）、物体力 **MAE 4.42 N**；H2O OOD 物体力 **MAE 3.88 N**；真机 cube/cylinder（8 人）仍恢复主要接触区与合理力幅。
6. **真机验证：** 自研透光亚克力 **instrumented cube/cylinder**，内相机测压–光响应得稠密接触/力真值；模型仅用仿真力监督训练。

## 开源边界（步骤 2.5，截至 2026-09-13）

| 已发布 | 备注 |
|--------|------|
| 训练 / 评测 / 预处理代码 | `train.py`、`evaluate_*.py`、`arctic_preprocess.py`、`force_sim/` 等 |
| HF 力仿真标注 | `arctic_force_simulations.zip`、`h2o_force_simulations.zip` |
| HF 真机数据集 | `egophi_dataset.zip` |
| 依赖数据 | 需自行下载 **ARCTIC**、**H2O**；另需 clone **HACO_RELEASE** |
| 预训练权重 | README 中 checkpoint 链接 **为空**（截至核查日） |
| 许可证 | README 声明 **MIT** |

**综合判定：** **部分开源** — 代码 + 仿真力标注 + 真机数据已发布；**预训练 checkpoint 链接待补**，复现需自训或等待官方权重。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-egophi.md`](../../wiki/entities/paper-egophi.md)
- 项目页：[`sources/sites/egophi-siplab.md`](../sites/egophi-siplab.md)
- 仓库：[`sources/repos/egophi.md`](../repos/egophi.md)
- 任务交叉：[手–物交互 / 模仿学习](../../wiki/methods/imitation-learning.md)、[Awesome Egocentric Vision](../../wiki/entities/awesome-egocentric-vision.md)
