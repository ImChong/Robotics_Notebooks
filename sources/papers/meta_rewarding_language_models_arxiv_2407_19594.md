# Meta-Rewarding Language Models：以元评委改进语言模型对齐

> 论文原始资料归档；以用户指定的 arXiv v2 为主要精读对象，以下是独立归纳而非原文转载。

- **标题：** Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge / 元奖励语言模型：通过 LLM 元评委实现自改进对齐
- **类型：** paper / language model alignment / self-improving judge / preference optimization
- **作者：** Tianhao Wu、Weizhe Yuan、Olga Golovneva、Jing Xu、Yuandong Tian、Jiantao Jiao、Jason Weston、Sainbayar Sukhbaatar
- **机构：** Meta FAIR、加州大学伯克利分校（University of California, Berkeley）、纽约大学（New York University）
- **arXiv：** [2407.19594v2](https://arxiv.org/abs/2407.19594v2)，2024-07-30（v1：2024-07-28）
- **原文：** [HTML 全文及附录](https://arxiv.org/html/2407.19594v2) / [PDF v2](https://arxiv.org/pdf/2407.19594v2)
- **正式发表：** [EMNLP 2025，ACL Anthology](https://aclanthology.org/2025.emnlp-main.583/)（2025-11，11537–11554，DOI: 10.18653/v1/2025.emnlp-main.583）；正式发表信息不等于用户给定 v2 的内容版本
- **作者公开资料：** [Tianhao Wu 的 Publications](https://thwu1.github.io/tianhaowu/publications/)（列论文与 EMNLP 页面，未列本论文实现仓库）
- **入库日期：** 2026-11-19
- **核对范围：** 通读 arXiv v2 HTML §1–§6、§2.1/§2.2 数据构造、§3.1–§3.5 实验/消融、附录 A.1–A.3（含 Table 6/7 和训练配置）；另核对 ACL Anthology 正式发表页及作者公开论文列表的资源入口。
- **通读：** 已完成

## 一句话说明

同一 Llama-3-8B-Instruct 模型在每轮分别扮演 Actor（回答）、Judge（打分）和 Meta-Judge（比较打分判断），用回答偏好和判断偏好训练模型；以显式提高评审质量来减轻单独自奖励的饱和问题。

## 核心机制（论文 §2）

1. **种子：** Llama-3-8B-Instruct 先用来自 Self-Rewarding 工作的 Evaluation Fine-Tuning（EFT）数据做 SFT，初始化可按 rubric 评分的模型；不能解释成从零、完全无人工数据。
2. **Actor 偏好对（§2.1）：** 每个提示采样 K=7 条回答；Judge 对每条回答采样 N=11 次评判，从有效的 1–5 分评分求均值；结合长度控制参数 ρ，从高分质量层中挑最短回答作 chosen，低分层中挑最长回答作 rejected。
3. **Judge 偏好对（§2.2）：** 选择对同一回答重复打分方差最高的案例；Meta-Judge 两两比较不同判断的质量，并通过交换位置、加权胜负与 Elo 拟合选出最佳/最差判断对。另过滤过长的 chosen judgment。
4. **更新：** 合并回答偏好与评审偏好，利用 DPO 优化同一模型；得到下一轮 M_(t+1)。三种角色是同一 checkpoint 的不同提示任务，不是三套独立训练模型。
5. **轮次边界（§3.1）：** 迭代 1、2 同时使用 Actor + Judge 偏好数据；迭代 3、4 **只用 Actor 偏好数据**，原因包括元评委评分偏差和饱和。不能写成四轮都持续训练元评委。

## 数据与指标（论文 §3）

- 每轮从固定的 20,000 条提示池抽取 5,000 条；K=7 → 约 35,000 条回答，N=11 次评分；回答采样 temperature=0.8、top_p=0.95。
- 论文附录：SFT 10 epochs、学习率 5e-8、batch 32；DPO 10 epochs、学习率 5e-6、β=0.1、batch 32；checkpoint 和长度阈值随迭代调整。
- **AlpacaEval 2**（length-controlled win rate）：种子 22.92%，Self-Rewarding + LC 四轮后 35.49%，Meta-Rewarding 四轮后 **39.44%**。此结果是论文报告值，未在本库重跑。
- **Arena-Hard**：种子 20.6%，论文方法四轮后 **29.1%**；均为语言指令基准，不是机器人任务成功率。
- **MT-Bench**（附录 Table 6）：种子总分 8.116，Meta-Rewarding 迭代 3 为 8.341、迭代 4 为 8.288；多轮对话收益并非单调。
- **Judge 对人类排序的 Spearman**（附录 Table 7）：种子 0.315，Meta-Rewarding 迭代 2 达 0.382、迭代 4 回到 0.326；与人工偏好的相关性并非持续上涨。

## 消融与关键局限（§3.5、§5）

- 长度控制是关键：ρ=0 时回答容易增长；Meta-Rewarding 在 Iter 4 使用 ρ=0.4 得到 39.44% LC win rate。
- Meta-Judge 偏好**更高分判断**的比例从 Iter 1 的 63.04% 增至 Iter 2 的 97.68%；位置偏差亦增长。Judge 的平均分从约 4.1 上移至 4.7 以上，导致评分饱和。
- 1–5 分量表大量打平，需重复取均值；Meta-Judge 的位置及高分偏好限制后续迭代，且对非自身生成回答的判断改善有限。
- 训练阶段已有来自人类排序的 EFT 数据，不能把「无需新的人工反馈」扩写成「无任何人工监督」。不提供长周期、跨模型迁移或机器人真机验证。

## 开源与复现边界

- 论文 v2、ACL Anthology 正式页和第一作者公开论文索引均未给出**明确归属于本论文的官方可运行训练/推理代码仓库**（核对日：2026-11-19）。
- 因无可核实的官方执行入口，知识节点不绘制臆造的源码运行时序图；仅以论文 Figure 1 / §2 绘制**算法流程图**。

## 对 wiki 的映射

- [Meta-Rewarding 独立论文详情](../../wiki/entities/paper-meta-rewarding-language-models.md)
- [Self-Rewarding 前作](../../wiki/entities/paper-self-rewarding-language-models.md)
- [RSI 纵深路线 Stage 3](../../roadmap/depth-rsi.md)

## 参考来源

- [arXiv v2 HTML（正文与附录，2024-07-30）](https://arxiv.org/html/2407.19594v2)
- [arXiv v2 PDF](https://arxiv.org/pdf/2407.19594v2)
- [EMNLP 2025 正式记录](https://aclanthology.org/2025.emnlp-main.583/)
- [第一作者论文索引](https://thwu1.github.io/tianhaowu/publications/)
