# Self-Rewarding Language Models（自奖励语言模型）

> 原始论文资料归档（ingest），按 arXiv v3 全文编译；不是对原论文的逐字转载。

- **标题：** Self-Rewarding Language Models / 自奖励语言模型
- **类型：** paper / LLM alignment / self-reward / preference optimization
- **作者：** Weizhe Yuan、Richard Yuanzhe Pang、Kyunghyun Cho、Xian Li、Sainbayar Sukhbaatar、Jing Xu、Jason Weston
- **机构：** Meta（Meta AI）、纽约大学（New York University）
- **会议：** ICML 2024（PMLR 235:57905–57923）
- **arXiv：** [2401.10020v3](https://arxiv.org/abs/2401.10020v3)（v3：2025-03-28；初版：2024-01-18）
- **原文：** [HTML 全文及附录](https://arxiv.org/html/2401.10020v3) / [PDF v3](https://arxiv.org/pdf/2401.10020v3) / [ICML 正式论文页](https://proceedings.mlr.press/v235/yuan24d.html)
- **官方作者页面：** [Richard Yuanzhe Pang 研究列表](https://yzpang.github.io/research/)（本论文条目仅列论文/BibTeX，未列配套实现）
- **入库日期：** 2026-11-15
- **核对范围：** arXiv v3 全文 §1–§6、实验表 1–4、附录 §A.1–§A.7；核对原文方法、初始化数据、训练与评分配置、消融实验及局限；核查 ICML 正式论文页与作者资料页上的资源入口。
- **通读：** 已完成

## 一句话说明

同一个 LLM 同时承担**生成候选回答**与**LLM-as-a-Judge 评估候选**的角色，将自评分变为偏好对，使用 Iterative DPO 更新参数，使生成能力与评判能力随迭代共同改善；属于有界自训练，并非机器人的物理奖励学习或开放式 RSI。

## 论文机制（§2）

1. **初始化（IFT + EFT）：** 在 Llama 2 70B 上用 Open Assistant 的 3,200 条高排名英文指令数据执行 IFT；再加入由人工排名筛选的 1,630 条评审训练例（EFT），其中评价输出为理由 + 0–5 分。
2. **生成新指令：** 主实验使用**固定**的 Llama 2-Chat 70B 以 8-shot Self-Instruct 风格生成提示；论文附录 A.5 另研究用当前模型生成指令，不能误将其写为主实验设置。
3. **候选与自评分：** 当前模型对一个新提示采样 N=4 个候选回答，对每个候选重复评分 3 次并取均值；五个评分维度为 relevance、coverage、usefulness、clarity、expertise。
4. **偏好数据：** 对每个提示选自评分最高/最低的回答构成 (x, winning, losing)；若最高与最低同分则弃对。得到 AIFT 数据，使用 DPO（beta=0.1）训练下一轮。
5. **模型序列：** M0=预训练底座；M1=IFT+EFT 的 SFT；M2=基于 M1 自奖励数据的 DPO；M3=基于 M2 自奖励数据的 DPO。没有单独冻结的外部 reward model。

## 数据与复现实验设置（§3.1）

- 底座 Llama 2 70B。IFT 3,200；EFT 训练 1,630、评估 541，EFT 与 IFT 不重叠。
- DPO 轮次：AIFT(M1) **3,964** 对；AIFT(M2) **6,942** 对。
- 生成四个回答：temperature 0.7、top-p 0.9；评审采样重复 3 次；DPO batch 16、beta 0.1、学习率由 1e-6 降到 1e-7。
- 评价：AlpacaEval 2.0（805 提示、GPT-4 判评、GPT-4 Turbo 对照），MT-Bench；偏好评审能力在 Open Assistant 人工排名测试集上测 pairwise accuracy、Spearman 和 Kendall tau；另有 50 指令的人类评审。
- **代码核验：** 原文、ICML 页面及共同作者研究列表未提供本论文明确的官方 GitHub 训练/推理入口。故**无可核验的官方源码运行时序图**；不能把第三方复现、通用 DPO 工具库算作论文实现。

## 原文报告结果（§3.2）

| 指标 | M1 | M2 | M3 | 口径 |
|---|---:|---:|---:|---|
| AlpacaEval 2.0 win rate | 9.94% | 15.38% | 20.44% | 对 GPT-4 Turbo，GPT-4 评审 |
| MT-Bench 总分 | 6.78 | 7.01 | 7.25 | 10 分制 |
| 与人工排序的 pairwise accuracy | 78.7% | 80.4% | 81.7% | 独立 Open Assistant 评审集 |

- IFT-only 的 SFT baseline 在 pairwise accuracy 上为 **65.1%**；加入 EFT 提升到 **78.7%**。
- 生成平均长度 **1,092 → 1,552 → 2,552**，可能混入长度偏好；数学与逻辑推理的增益相对有限。
- 附录 A.3：移除 EFT 会使评分格式不稳定，偏好对数量大幅减少；附录 A.4：单独加入 11,254 个“5 分”正样本进行 SFT 没得到同样收益，说明偏好对训练是关键一环。

## 局限与工程解读

- 只在一个 70B 底座与 M1–M3 三轮序列上验证；没有长时间自我强化的可持续性证据。
- 评分与最终部分评估都依赖语言模型，仍可能存在 reward hacking、同源偏差、长度偏差；安全评估尚未开展。
- 属于 LLM 指令遵循任务，**没有机器人策略控制、真机交互或物理验证器的实验证据**。
- 对机器人研发的可迁移启发是“自生成偏好 → 训练 → 外部独立验收”设计，但真实机器人奖励必须另外设计可审计的环境/任务指标；此项为工程延伸，不是论文声称。

## 对 wiki 的映射

- [Self-Rewarding 独立论文实体](../../wiki/entities/paper-self-rewarding-language-models.md)
- [RSI 全谱系综述](../../wiki/entities/paper-rsi-survey-2607-07663.md)（training-time self-iteration 与 self-evaluation 的交叉）
- [RSI 纵深路线 Stage 3](../../roadmap/depth-rsi.md)（评判者与外部验证锚）
- [Self-Refine](../../wiki/entities/paper-self-refine-2303-17651.md)（推理期改写、无权重更新，对照区别）

## 参考来源

- [arXiv v3 全文（2025-03-28）](https://arxiv.org/html/2401.10020v3)
- [ICML 2024 PMLR 正式论文页](https://proceedings.mlr.press/v235/yuan24d.html)
- [共同作者公开研究清单](https://yzpang.github.io/research/)
