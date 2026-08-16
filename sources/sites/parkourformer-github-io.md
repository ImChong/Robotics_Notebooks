# parkourformer.github.io（ParkourFormer 项目页）

- **标题：** ParkourFormer — Integrating Predictive Supervision and Sequence Modeling into Parkour Locomotion
- **类型：** site / project-page
- **URL：** <https://mronaldo-gif.github.io/parkourformer.github.io/>
- **站点仓：** <https://github.com/MRonaldo-gif/parkourformer.github.io>（仅 github.io 主页，非训练代码）
- **arXiv：** <https://arxiv.org/abs/2605.25782>
- **机构：** HKUST-GZ；CLAI-LAB / CL-TECH；华南农业大学；广东工业大学
- **平台：** Unitree G1（29 DoF）
- **配套论文归档：** [`sources/papers/parkourformer_arxiv_2605_25782.md`](../papers/parkourformer_arxiv_2605_25782.md)
- **入库日期：** 2026-08-16

## 一句话摘要

HKUST-GZ 等单位的人形跑酷项目页：用 **Transformer + 未来两步本体监督** 把策略从 reactive 映射改成 future-conditioned Seq2Seq，在九类地形上以**单一策略**跑仿真 L1–L9 与真机楼梯/平台/缺口。

## 开源状态（步骤 2.5，截至 2026-08-16）

| 资源 | 状态 |
|------|------|
| 项目页 + 真机/仿真视频 + BibTeX | **已发布** |
| arXiv PDF/HTML | **已发布**（2605.25782） |
| 训练 / 推理代码 / 权重 | **未列出**（页上无 Code 按钮；GitHub 仅 `parkourformer.github.io` 站点仓） |

**结论：确认未开源（无可运行官方实现）。** wiki「源码运行时序图」标 **不适用**。

## 公开信息要点

- **副标题：** Integrating Predictive Supervision and Sequence Modeling into Parkour Locomotion。
- **叙事主线：** 当前状态 cross-attention 查询历史 → 预测头预报短时域本体 → 预测未来与时序特征融合出动作。
- **摘要数字：** 高难地形平均穿越成功率 **93.85%**，相对 MLP / MoE-MLP / vanilla Transformer 最高约 **+47.12 pt**。
- **仿真说明：** 训练含 9 类地形 × 9 级难度（L1–L9）；页上展示 L9 最高难度片段。
- **BibTeX：** `@article{mai2026parkourformer, ... journal={arXiv preprint arXiv:2605.25782}}`。

## 为何值得保留

- 与 [Hiking in the Wild](../../wiki/entities/paper-hiking-in-the-wild.md) 同用 Project Instinct MuJoCo，但把贡献从「感知 + AMP + MoE」转到 **显式未来监督 + query 历史**。
- 消融把「下楼靠 MSE 监督、缺口靠 RGB-D」拆开，比只报 Transformer 容量更可操作。
- 对照 [PHP](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md) / [LightLP](../../wiki/entities/paper-light-loco-parkour.md)：不走技能链或种子扩张，押 **单网 + 短时域 foresight**。

## 关联资料

- 论文归档：[`sources/papers/parkourformer_arxiv_2605_25782.md`](../papers/parkourformer_arxiv_2605_25782.md)
- 沉淀实体：[`wiki/entities/paper-parkourformer.md`](../../wiki/entities/paper-parkourformer.md)
