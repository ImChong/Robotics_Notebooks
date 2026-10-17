# Steertuning 项目页（steertuning.github.io）

> 来源归档

- **标题：** Steertuning Policies from a Few Human Videos
- **类型：** site（项目页）
- **URL：** <https://steertuning.github.io/>
- **论文：** <https://arxiv.org/abs/2610.18711>（归档：[steertuning_arxiv_2610_18711.md](../papers/steertuning_arxiv_2610_18711.md)）
- **代码：** 页面有 Code / Model 按钮但无 URL，未列 GitHub / Hugging Face；摘要称将发布数据集、权重、代码
- **入库日期：** 2026-10-17
- **一句话说明：** 5 条人类示范、约 2 分钟微调、总流程不到 5 分钟（页面表述），12 任务 / 12 新环境 71%。

## 页面结构

| 区块 | 内容要点 |
|------|----------|
| Teaser / Method | 第一人称数据处理流水线（Qwen 描述、Gemini 定位、SAM 掩码、去除手臂） |
| Results | 71%（12 任务 12 环境）；实验室三任务 83%；物体泛化 Flower 7/10 与 6/10（换水壶）、Toy 10/10 与 9/10 |
| Ablations | 微调、steering 强度、示范选择，仅给近似曲线 |
| Full-stack | YOR 底盘 + Wuji 手；取包、丢杯、擦桌 |

## 开源核查结论

截至 2026-10-17：Code / Model 按钮无链接，未发布。

## 对 wiki 的映射

- [wiki/entities/paper-steertuning.md](../../wiki/entities/paper-steertuning.md)
- [sources/papers/steertuning_arxiv_2610_18711.md](../papers/steertuning_arxiv_2610_18711.md)
