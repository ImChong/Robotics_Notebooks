# One-Step Euler FM（arXiv:2610.30735）

> 来源归档（paper）

- **标题：** When One Step Is Enough: An Empirical Study of Euler Integration in Flow-Matching Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.30735>
- **PDF：** <https://arxiv.org/pdf/2610.30735>
- **作者：** Shivakanth Sujit, Luca Nunziante, Rousslan Fernand Julien Dossa, Kai Arulkumaran
- **arXiv 评论栏：** Accepted at the 10th Conference on Robot Learning (CoRL 2026)

- **入库日期：** 2026-10-26
- **一句话说明：** 何时一步就够：Flow Matching 策略中欧拉积分的实证研究。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** Flow Matching Policy 通常需要通过数值积分生成动作。增加积分步数会提高推理开销，但更多步数不一定带来相应的控制收益。
3. **周报所述创新点：** 系统研究欧拉积分步数对生成式机器人策略的影响，重点分析单步积分在不同条件下的适用性。论文已被 CoRL 2026 接收。
4. **arXiv 摘要（英文节选）：** Flow matching is now the dominant approach for vision-language action (VLA) robot policies, thanks to its expressiveness and simple training objective. However, inference for standard flow matching requires iteratively solving an ODE, limiting the control frequency at deployment. While several works have proposed novel methods to allow for single-shot action generation, we ask whether such additional objectives are actually necessary.

**对 wiki 的映射**

- [paper-one-step-euler-integration-flow-matching-policies](../../wiki/entities/paper-one-step-euler-integration-flow-matching-policies.md)
