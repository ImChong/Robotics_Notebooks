# ame2（Kitjesen/ame2）

> 来源归档（ingest · 步骤 2.5 社区复现）

- **标题：** AME-2 Standalone PyTorch Implementation for ANYmal-D
- **类型：** repo（**非官方**社区复现）
- **组织：** Kitjesen（个人）
- **代码：** <https://github.com/Kitjesen/ame2>
- **论文：** [AME-2（arXiv:2601.08485）](../papers/humanoid_pnb_ame-2-agile-and-generalized-legged-locomotion-vi.md) · [AME-1 前作（arXiv:2506.09588）](../papers/ame_arxiv_2506_09588.md)
- **项目页：** [ame-2-leggedrobotics.md](../sites/ame-2-leggedrobotics.md)
- **入库日期：** 2026-09-10
- **一句话说明：** 社区 **standalone PyTorch** 复现 AME-2 的 **AME-2 encoder + 神经映射 + Teacher–Student** 管线；目标平台 **ANYmal-D**（非论文官方 Isaac Gym 栈）；README 标明为 arXiv:2601.08485 独立实现。

## 开源状态（步骤 2.5，截至 2026-09-10）

| 项 | 状态 |
|----|------|
| 与 ETH 官方关系 | **非官方** — [AME-2 项目页](../sites/ame-2-leggedrobotics.md) **无** GitHub 链接；论文作者未发布训练/部署仓库 |
| 可运行入口 | **有（社区）** — 仓库含训练/推理脚本与模块划分（PyTorch；ANYmal-D） |
| 预训练权重 | 见仓库 README / Releases（入库日未逐项镜像） |
| 平台差异 | 论文为 **Isaac Gym + RSL-RL** 官方栈；本仓为 **独立 PyTorch** 社区实现 |
| 对照 | [SII-FUSC/AME_Locomotion](ame_locomotion_sii_fusc.md) 为 AME-1 **G1 + Isaac Lab** 社区复现 |

判定：**社区已开源（非官方 ANYmal-D 复现）**；引用 AME-2 方法时请区分 **ETH 官方未发布代码** 与本仓实现边界。

## 对 wiki 的映射

- [`wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md`](../../wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md)
- [`wiki/entities/paper-ame-attention-based-map-encoding.md`](../../wiki/entities/paper-ame-attention-based-map-encoding.md)
- [`sources/sites/ame-2-leggedrobotics.md`](../sites/ame-2-leggedrobotics.md)
