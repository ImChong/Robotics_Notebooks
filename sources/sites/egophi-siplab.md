# EgoPHI 项目页（siplab.org/projects/EgoPHI）

> 来源归档

- **标题：** EgoPHI: Estimating 3D Hand-Object Contact and Force from Egocentric Vision
- **类型：** site / project-page
- **URL：** <https://siplab.org/projects/EgoPHI>
- **论文：** <https://arxiv.org/abs/2608.13014>
- **代码：** <https://github.com/eth-siplab/EgoPHI>
- **数据：** <https://huggingface.co/datasets/eth-siplab/EgoPHI>
- **机构：** 苏黎世联邦理工（ETH Zürich）；Sensing, Interaction & Perception Lab（SIPLAB）
- **出处：** ECCV 2026
- **入库日期：** 2026-09-13
- **一句话说明：** 官方项目站：三阶段 InteractionGNN 管线说明、ARCTIC/H2O/真机定量表与交互式 3D 力可视化。

## 开源核查（步骤 2.5，截至 2026-09-13）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | 是 → GitHub `eth-siplab/EgoPHI` |
| 项目页是否链到数据 | 是 → Hugging Face `eth-siplab/EgoPHI` |
| 仓内可运行训练入口 | **是** — `train.py` / `evaluate_*.py` / 预处理脚本 |
| 预训练权重 | README checkpoint 链接 **为空** |
| 综合判定 | **部分开源**（见 [`sources/repos/egophi.md`](../repos/egophi.md)） |

## 公开信息要点

- **核心贡献：** 从单目 ego RGB + 已知物体几何，预测双手与关节刚体物体 mesh 上的 **稠密 3D 接触与力**（非仅 2D 热图）。
- **IRM：** 先细化物体 3D 位姿再预测力，应对手部重度遮挡。
- **仿真监督：** SOFA 物理管线为 ARCTIC 增补 per-vertex 力标注。
- **基线：** PressureVision（2D）、HACO（3D mesh，多数据集预训练）；EgoPHI 仅 ARCTIC 训练仍优于或保持 OOD 竞争力。
- **真机：** 透光亚克力 cube/cylinder + 8 参与者多样触摸/抓取；仿真训练仍恢复主要接触区。

## 关联资料

- 论文摘录：[`sources/papers/egophi_arxiv_2608_13014.md`](../papers/egophi_arxiv_2608_13014.md)
- 仓库归档：[`sources/repos/egophi.md`](../repos/egophi.md)
- Wiki 实体：[`wiki/entities/paper-egophi.md`](../../wiki/entities/paper-egophi.md)
