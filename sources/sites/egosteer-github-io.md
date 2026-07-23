# EgoSteer 项目页（egosteer.github.io）

- **标题：** EgoSteer: A Full-Stack System Towards Steerable Dexterous Manipulation from Egocentric Videos
- **类型：** site / project-page
- **URL：** <https://egosteer.github.io/>
- **论文：** [arXiv:2607.09701](https://arxiv.org/abs/2607.09701)（PDF：<https://arxiv.org/pdf/2607.09701>）
- **代码：** [`egosteer/egosteer`](https://github.com/egosteer/egosteer) · [`egosteer/egosmith`](https://github.com/egosteer/egosmith) · [`egosteer/robot-stack`](https://github.com/egosteer/robot-stack)（Apache-2.0）
- **权重：** <https://huggingface.co/EgoSteer/models>（`EgoSteer-3B-Base` / `EgoSteer-3B-RealMan`）
- **数据入口：** <https://huggingface.co/datasets/egosteer>（截至 2026-07-23 组织下公开 datasets 仍为 0；项目页写处理后数据「几周内 / 几天内」开源）
- **入库日期：** 2026-07-23
- **配套论文归档：** [`sources/papers/egosteer_arxiv_2607_09701.md`](../papers/egosteer_arxiv_2607_09701.md)

## 一句话摘要

北京大学 / PKU–PsiBot / UPenn 提出的 **可操控（steerable）双灵巧手全栈**：EgoSmith 策展 **9.6K h** egocentric 预训练数据，Unified Robot Stack 统一遥操作与 HITL DAgger，EgoSteer 以世界模型增强的 flow-VLA 在 **40+** 自由语言任务上展示失败恢复、灵巧与组合泛化，并支持双具身长程 few-shot。

## 开源状态（步骤 2.5，截至 2026-07-23）

| 项 | 状态 |
|----|------|
| 项目页 Code badge | **有** → GitHub org `egosteer` 三仓库 |
| Hugging Face 权重 | **有**（Base + RealMan，Apache-2.0） |
| 处理后全量数据集 | **待发布**（HF datasets 公开数 = 0；项目页承诺近期开源） |
| 示例训练数据 | **有**（主仓 Google Drive `example_data.zip`） |

结论：**部分开源** — 训练/推理/遥操作代码与模型权重可获取；大规模处理后人/机数据集以项目页后续发布为准。

## 页面内容要点

- **三模块：** EgoSmith / Unified Robot Stack / EgoSteer（世界模型增强 VLA）。
- **Demo Gallery：** Seen / Failure Recovery / Dexterity / Compositional / Unseen；Few-shot 长程（折盒、开蛋糕盒、叠杯、制冰可乐、手机包装等）。
- **数据叙事：** 12 源 egocentric → 9.6K h；真机遥操作数据集将开源。
- **BibTeX：** `arXiv:2607.09701`，通讯 Yuanpei Chen / Yaodong Yang。

## 关联资料

- 论文归档：[`sources/papers/egosteer_arxiv_2607_09701.md`](../papers/egosteer_arxiv_2607_09701.md)
- 仓库归档：[`egosteer`](../repos/egosteer.md) · [`egosmith`](../repos/egosmith.md) · [`robot-stack`](../repos/egosteer-robot-stack.md)
- wiki：[`wiki/entities/paper-egosteer.md`](../../wiki/entities/paper-egosteer.md)
