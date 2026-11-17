# Reasoning with Sampling：采样即推理（arXiv:2510.14901）

> 来源归档（paper）

- **标题：** Reasoning with Sampling: Your Base Model is Smarter Than You Think（通过采样进行推理：基础模型比你想象的更聪明）
- **类型：** paper
- **作者：** Aayush Karan、Yilun Du
- **机构：** 哈佛大学（Harvard University）
- **发表信息：** arXiv:2510.14901v1（2025-10-16）；ICLR 2026 会议论文（ICLR Proceedings 收录）
- **论文：** <https://arxiv.org/abs/2510.14901>
- **全文 HTML：** <https://arxiv.org/html/2510.14901v1>
- **PDF：** <https://arxiv.org/pdf/2510.14901>
- **项目页：** <https://aakaran.github.io/reasoning_with_sampling/>
- **官方代码：** <https://github.com/aakaran/reasoning-with-sampling>
- **入库日期：** 2026-11-17
- **一句话说明：** 在不训练和不使用外部答案验证器的条件下，利用自回归 Metropolis–Hastings 近似采样整体序列的幂分布，提升基础语言模型单次推理性能并保持多样性。

## 原文摘录与核验

1. **动机（`1、`4.1）：** RL 后训练输出倾向落在基础模型本来就赋予高概率的区域，作者考察推理时对整个序列分布做 sharpening 能否部分替代 RL。
   - **对 wiki 的映射：** [方法目标、与 GRPO 对比](../../wiki/entities/paper-reasoning-with-sampling.md)。
2. **分布区别（`4.1、命题 1）：** 序列级 `p(x_{0:T})^\alpha` 与逐 token 温度 `p(x_t\mid x_{<t})^\alpha` 不等价；前者的边际权重含未来路径的“幂后求和”，后者则是“求和后取幂”。
   - **对 wiki 的映射：** [核心原理](../../wiki/entities/paper-reasoning-with-sampling.md)。
3. **算法（`4.2–4.3、Algorithm 1）：** 对已有前缀扩展 `B` 个 token；反复均匀选重采样起点，用提议模型生成新后缀，再按目标分布与正反 proposal 概率的 MH 比率接受或拒绝；逐块增长序列以降低长序列 MCMC 混合难度。
   - **对 wiki 的映射：** [流程总览及源码时序图](../../wiki/entities/paper-reasoning-with-sampling.md)。
4. **实验（`5、Table 1）：** Qwen2.5-Math-7B 的 MATH500：基础 49.6%、温度采样 69.0%、Power Sampling 74.8%、GRPO 78.5%；HumanEval：32.9%、51.2%、57.3%、53.7%。此数字为作者报告，非入库者重跑。
   - **对 wiki 的映射：** [实验与评测](../../wiki/entities/paper-reasoning-with-sampling.md)。
5. **开销与边界（`5.3、Figure 6、附录 A.1–A.3）：** 最大生成 3072 token、16 块（`B=192`）、`\alpha=4`、常用 `N_{\rm MCMC}=10`；MATH500 平均生成长度 679 时作者估算重复生成 token 量约标准解码的 8.84 倍。算法并不表示已验证实时机器人部署或无开销。

## 复现资料核对

- 项目页公开 Paper、arXiv 和 Code 入口，代码指向作者的 `aakaran/reasoning-with-sampling`。
- 已静态阅读官方 `README.md`、`llm_experiments/power_samp_math.py`、`llm_experiments/power_samp_utils.py`（`AutoregressiveSampler`、`naive_temp`、`mcmc_power_samp`）以及 `llm_experiments/eval_math.py`。
- 官方提供 MATH500 数据、HumanEval / GPQA / AlpacaEval 采样脚本与评测脚本；未实际运行 GPU 生成或独立复现指标。
- 本论文研究的是语言模型测试时推理分布，不是机器人控制策略的 sim2real 验证。

## 对 wiki 的映射

- [Reasoning with Sampling 独立论文节点](../../wiki/entities/paper-reasoning-with-sampling.md)
- [AI Agent 评测（pass@k 对照）](../../wiki/concepts/ai-agent-evaluation.md)

- **核对范围：** 通读 arXiv:2510.14901v1 官方 HTML 正文 `1–7、主要表格与图示说明、附录 A.1–A.3（理论、pass@k 与样例）；对照作者项目页及上述四个官方代码文件/README。只做静态文献与源码核验，未运行实验。
- **通读：** 已完成
