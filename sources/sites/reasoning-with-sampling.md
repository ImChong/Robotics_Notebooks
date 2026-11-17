# Reasoning with Sampling 官方项目页

> 来源归档（site）

- **标题：** Reasoning with Sampling: Your Base Model is Smarter Than You Think
- **类型：** site
- **链接：** https://aakaran.github.io/reasoning_with_sampling/
- **论文：** <https://arxiv.org/abs/2510.14901>
- **代码：** <https://github.com/aakaran/reasoning-with-sampling>
- **入库日期：** 2026-11-17
- **作者与机构：** Aayush Karan、Yilun Du；哈佛大学（Harvard University）
- **一句话说明：** 官方项目页解释序列幂分布、分块自回归 MCMC、与 GRPO 的 single-shot / pass@k 对比及推理开销。
- **沉淀到 wiki：** [论文实体](../../wiki/entities/paper-reasoning-with-sampling.md)

## 开放状态与边界

- **已开源：** 项目页 Code 链接指向公开 PyTorch 仓库；包含 MATH500、HumanEval、GPQA Diamond 和 AlpacaEval 2.0 实验入口。
- **量化结果：** 官网报告与 arXiv v1 相符；single-shot 相比基础模型增强并不代表计算成本相同或任何领域都胜过 RL。
- **外推提醒：** 官网没有声称已完成真机控制、机器人任务评测或实时机载测试。

- **核对范围：** 通读官方项目页的 Abstract、Power Distributions、Autoregressive MCMC、Single-shot Reasoning、Diversity/pass@k、Test-time Scaling 和 Implications；检查 Paper、arXiv、Code 链接。
- **通读：** 已完成
