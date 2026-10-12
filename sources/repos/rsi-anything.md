# Gen-Verse/RSI-Anything

> 来源归档（官方代码仓）

- **仓库：** <https://github.com/Gen-Verse/RSI-Anything>
- **类型：** workflow runner / 任务适配器
- **许可证：** Apache-2.0
- **关联论文：** [RSI-Anything](../papers/rsi_anything_arxiv_2610_13912.md)（arXiv:2610.13912）
- **项目页：** [PhAI Labs 项目稿](../sites/rsi-anything.md)
- **入库日期：** 2026-10-12
- **一句话说明：** 实现以预测式世界模型驱动候选选择和迭代保留的递归自我改进流程。

## 仓库边界

仓库公开共享 CLI workflow runner、任务适配器、配置、文档和代码修复/GUI 任务实现。README 概括主环为 **Propose → Predict → Select → Evaluate → Learn → Retain**；不同任务的具体训练、行动候选与验证逻辑由 adapter 承担，因此这个流程不应误读成所有科学领域共用同一种执行器。

`docs/workflow.md` 中代码修复完整流程包括资产检查、初始/语义/扩展/incumbent world 更新和评估；GUI 流程包含模型获取、环境预检、适配与评估。实验室规划、反应优化和免疫假设等论文结果不能仅凭存在共享 runner 推断为完整公开的实验执行代码。

## 复现条件

官方复现文档记录原始环境为 Linux、Python 3.10.12、NVIDIA A800 80GB。两个基础模型约 21.4 GB，此外还需分别获取模型及任务资产、依赖和数据。代码公开不代表开箱即可复现论文全部实验；应按仓库对应任务文档逐项核对。

## 对 wiki 的映射

- [RSI-Anything 实体页](../../wiki/entities/paper-rsi-anything.md) — 论文结果、运行时序及复现边界。
- [递归自我改进](../../wiki/concepts/recursive-self-improvement.md) — 将其作为有界、可验证迭代案例。
