# Mind the Context: Continual Learning of Socially Appropriate Robot Actions（arXiv:2608.13448）

> 来源归档（ingest）

- **标题：** Mind the Context: Continual Learning of Socially Appropriate Robot Actions via Environmental-Social Disentanglement
- **缩写 / 框架：** **EDD**（Explicit Disentanglement Dual-Branch）
- **类型：** paper / social-robot / continual-learning / hri
- **arXiv：** <https://arxiv.org/abs/2608.13448>
- **会议：** IROS 2026（扩展版）
- **代码：** <https://github.com/Cambridge-AFAR/Mind-the-Context>（归档见 [`sources/repos/mind-the-context.md`](../repos/mind-the-context.md)）
- **作者：** Rafal Robert Karpinski、Fethiye Irmak Dogan、Nikhil Churamani、Yiming Luo、Maartje M.A. de Graaf、Davide Dell'Anna、Hatice Gunes
- **机构：** 剑桥大学（Cambridge AFAR）等
- **入库日期：** 2026-08-18
- **一句话说明：** domain-incremental 持续学习社交适当动作：双分支拆开环境线索与社会主体线索，replay 缓解遗忘。

## 开源状态（步骤 2.5）

- **无独立项目页**；以 GitHub 为准。
- **仓库核查（2026-08-18）：** [Cambridge-AFAR/Mind-the-Context](https://github.com/Cambridge-AFAR/Mind-the-Context) 默认分支 `iros2026`；含 `models/`、`experiments/training.ipynb`、`evaluation.ipynb`；MANNERSDB+ / OFFICE-MANNERSDB 需自备，仓内无 LICENSE。
- **结论：** **已开源、可辨识训练/评测入口**（notebook）；数据集不随仓。

## 摘录

相似家具布局在客厅与会议室意味着完全不同的「能不能搭话 / 清扫」。EDD 显式拆 environmental vs social-agent knowledge。评测跨室内域（客厅、会议室、办公室、走廊），优于多种 CL 基线；另有解耦策略与域顺序消融。

**对 wiki 的映射：** [`wiki/entities/paper-mind-the-context.md`](../../wiki/entities/paper-mind-the-context.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（notebook 可辨识）
