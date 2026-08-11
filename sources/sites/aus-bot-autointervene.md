# AutoIntervene 项目页（aus.bot）

> 来源归档

- **标题：** AutoIntervene — Calibrated intervention for action-chunking policies
- **类型：** site / project-page
- **URL：** <https://aus.bot/AutoIntervene/>
- **研究站入口：** <https://aus.bot/research/autointervene/>
- **论文：** <https://arxiv.org/abs/2608.07065> — 归档见 [`sources/papers/autointervene_arxiv_2608_07065.md`](../papers/autointervene_arxiv_2608_07065.md)
- **代码：** 截至 **2026-08-11** 项目页 **未列** 训练/推理 GitHub；`https://github.com/123qwedsa123/AutoIntervene` 仅为静态项目页镜像
- **机构：** PAIR Lab / The University of Sydney；Vanderbilt University
- **入库日期：** 2026-08-11
- **一句话说明：** AutoIntervene 官方项目站：双向校准接管方法叙事、九项双臂真机视频与成功率/操作员时间表。

## 开源核查（步骤 2.5，截至 2026-08-11）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **否** — 入口为论文、方法说明、实验视频与结果表 |
| GitHub `123qwedsa123/AutoIntervene` | **项目页静态仓**（`index.html` / `styles.css` / `assets`），非训练栈 |
| 项目页是否链到数据 / 权重 | **否** |
| 综合判定 | **确认未开源**（无可运行训练/部署实现） |

## 页面要点

- Hero：选择性在 visuomotor 策略与操作员间转移控制；针对性恢复成为下一轮监督。
- Method：Construct → Evaluate → Transfer → Adapt；phase-local vs global support；held-out 分位数校准阈值。
- 九任务：Peg Disassembly、Potato Transfer、Towel Folding/Bagging、Lidded Box Packing、Plant Sorting、Towel/Two-Towel Box Packing、Towels-and-Cable Bagging。
- 主结果（七任务 avg）：Initial 30.9% → AutoIntervene R2 **80.0%**（操作员 Δt 122.9 s）优于 Human R2 68.6% 与 Additional Full Data 56.0%。
- 兼容 ACT / Diffusion Policy / Flow Matching。

## 关联资料

- 论文摘录：[`sources/papers/autointervene_arxiv_2608_07065.md`](../papers/autointervene_arxiv_2608_07065.md)
- Wiki 实体：[`wiki/entities/paper-autointervene.md`](../../wiki/entities/paper-autointervene.md)
