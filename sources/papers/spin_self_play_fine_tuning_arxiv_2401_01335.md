# Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models（SPIN，自博弈微调）

> 一手论文归档（ingest）

- **英文标题：** Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
- **中文标题：** 自博弈微调使弱语言模型变强
- **作者：** Zixiang Chen、Yihe Deng、Huizhuo Yuan、Kaixuan Ji、Quanquan Gu
- **机构：** 加州大学洛杉矶分校（University of California, Los Angeles，UCLA）
- **会议：** ICML 2024
- **论文 / 版本：** https://arxiv.org/abs/2401.01335v3 （2024-06-14 修订）；https://arxiv.org/pdf/2401.01335v3
- **项目：** https://uclaml.github.io/SPIN/ （[项目页归档](../sites/spin-project.md)）
- **官方代码：** https://github.com/uclaml/SPIN （[代码归档](../repos/spin-uclaml.md)）
- **入库日期：** 2026-11-17
- **核对范围：** 论文 v3 正文 ``1–7、主要公式（`4、`5）、Table 1 与 Figure 2/3、附录中的理论和实验补充；并交叉核对官方项目页及仓库 README、数据生成/转换/训练/损失源码。
- **通读：** 已完成
- **一句话说明：** 在 SFT 参考回答与上一轮模型自生成回答之间构造训练对，用冻结的上一轮模型作为隐式判别参考，迭代优化语言模型，不额外收集新的人类偏好标注。

## 摘录 1：研究问题与方法（``1–4）

- 标准监督微调（SFT）过分依赖初始示范质量；SPIN 试图在已有 SFT 数据集上让模型继续改进。
- 第 t 轮由当前模型针对同一提示 x 生成回答 y'，将 SFT 数据中的参考回答 y 当作目标分布样本；下一轮模型要区分 y 与 y'。
- 优化采用相对上一轮模型的回答对数概率差（正例 y 与负例 y'），形态类似 DPO，但负例由模型自己生成；不另训练外部 reward model。
- SPIN 的“博弈”是迭代模型与之前模型的分布判别，不是双机器人对抗、环境 RL 或额外人类偏好比较。

**对 wiki 的映射：** [SPIN 唯一论文实体](../../wiki/entities/paper-spin-self-play-fine-tuning.md) 的核心机制与流程图。

## 摘录 2：理论与边界（`5）

- 论文从自博弈视角导出基于 sigmoid / logistic 的目标，并讨论模型输出分布何时匹配目标数据分布。
- 理论平衡点成立依赖函数类、损失和可表达性等假设；不能把它误读成有限训练轮数一定单调增长、自动超越全部训练数据质量。
- 该方法始终使用原有 SFT 参考数据作为分布锚点；“无需额外偏好标注”不是“从无监督起步”。

**对 wiki 的映射：** [SPIN 理论边界与局限](../../wiki/entities/paper-spin-self-play-fine-tuning.md)。

## 摘录 3：实验与对照（`6 / Table 1 / Figure 2–3）

- 基座是基于 Mistral-7B 的 Zephyr SFT 模型；SFT 指令样本来自 UltraChat 的 50k prompts，后续迭代可合并使用自生成配对。
- 作者报告 Open LLM Leaderboard 六任务平均分：SFT 起点 58.14；SPIN iter0 60.80；iter1 62.12；iter2 62.97；iter3 63.16。
- 使用外部偏好样本的 Zephyr DPO 参照为 61.31；SPIN iter3 更高，但比较受模型、配方和评测版本条件约束。
- MT-Bench 由 SFT 起点 5.94 到 iter3 6.78；不是所有单项都提升（例如 MMLU 在原文表格中有波动）。
- 实验是在语言模型基准上报告；论文没有提供真机控制、具身策略或 Sim2Real 实验。

**对 wiki 的映射：** [SPIN 评测与可操作结论](../../wiki/entities/paper-spin-self-play-fine-tuning.md)。

## 复现注意事项（官方 README）

- 2024-04-04 作者更正过已发布数据集；复现须使用更正后的 Hugging Face 数据。
- Alignment Handbook 后续更新了 SFT checkpoint；使用官方发布的 SPIN 配对数据时，要匹配 README 所指旧版模型 revision。
- 官方复现脚本基于 Accelerate + DeepSpeed ZeRO-3，多卡 A100 80GB；未在本次 ingest 实际运行 GPU 训练。