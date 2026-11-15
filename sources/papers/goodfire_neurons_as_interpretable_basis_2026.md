# Goodfire：Neurons as an Interpretable Basis（神经元作为可解释的计算基底）

> 原始技术研究归档。与 Nick Cammarata 的《Neural Networks Are Already Clean》属于同一研究脉络，**并非**同一篇论文；正式结论仅限作者报告的模型、实验与激活区间。

- **标题：** Neurons as an Interpretable Basis / 神经元作为可解释的计算基底
- **作者：** Curt Tigges、Matthew Kowal、Thomas McGrath、Owen Lewis、Nick Cammarata（Goodfire）
- **发表日期：** 2026-11-10
- **类型：** Goodfire 官方技术研究文章（官方页面给出 BibTeX，未提供可核对的 arXiv ID 或 DOI）
- **官方研究全文：** <https://www.goodfire.com/research/neurons-as-an-interpretable-basis>
- **全文 Markdown：** <https://www.goodfire.com/research/neurons-as-an-interpretable-basis/article.md>
- **交互神经元地图：** <https://www.goodfire.com/research/public_neuron_atlas>（需要 JavaScript）
- **前置研究随笔：** <https://already-clean.goodfire.com/>
- **核对范围：** 官方全文的 Abstract、Introduction、SwiGLU neuron structure、Neuron Sparsity and Effective Range、Understanding Neuron Computations、Subcircuits/Modes、random-direction baselines、Evaluating Neuron Descriptions、Conclusion/Future Work；附录中的细节及全部图表尚未逐项核完。
- **通读：** 未完成
- **入库日期：** 2026-11-15

## 研究对象与方法

作者在 **Qwen3.6-35B-A3B** 的 MoE MLP 中，将神经元定义为 SwiGLU 中间层的计算单元，研究单个神经元的**正/负极（pole）**、经 expert router 加权的激活、上下游连接和因果影响。

1. **有效稀疏性：** 逐 token 保留每层 4,096 个 routed neuron 中按激活绝对值加权选出的 top-320，其余置零（消融 92.2%）；共享 expert 不在本次消融范围。
2. **计算角色：** 在因果意义更显著的强激活区间追踪 gate/up 输入与下游 reader，提取子电路，再按计算语义归并为少量 mode。作者明确认为“一个神经元只有一个意义”并不是必要条件。
3. **可检验解释：** 由研究 Agent 生成功能描述，再做成对激活识别、mode 归类、逐 token 激活模拟和反向合成输入；通过真实模型激活及干预验证描述，而不只看标签是否通顺。
4. **对照：** 与随机方向进行下游读取结构、语义集中度及 KL 功能恢复比较，检验原生 neuron basis 是否确有特殊性。

## 原文可核对的实验数字

| 指标 | 作者报告 | 正确的解读范围 |
|---|---|---|
| routed neuron 稀疏消融 | 消融 92.2% 的 routed neuron 激活，语言模型 loss 上升 2.9% | 不包括共享 experts；并非删去模型 92.2% 权重，也不代表所有推理任务无损 |
| 激活检测 | 依据描述区分激活文本与随机对照，约 95% 准确率 | 针对作者设定的**成对辨别**，并非跨场景解释成功率 |
| 模拟因果效应 | 在强激活范围内恢复约 73% 的**可恢复消融 KL 损害上限** | 是相对 perfect-reader ceiling 的比例，非“73% 的神经元全部理解” |
| 输入合成 | 128 个 neuron poles 中 120 个超过 100 亿 token 自然语料的最大激活 | 带激活反馈的迭代合成实验，不能直接等同于零样本控制能力 |

## 方法边界与复现状态

- 作者承认弱激活、Q3 激活、跨模型与跨任务泛化、神经元功能如何在训练中形成等仍待研究；从单神经元扩展到整体电路和模型安全也尚未解决。
- **对照基线重要：** 随机方向并非完全不可解释，但原生神经元在作者实验中更易由少量单元恢复功能，且与特定下游 reader 的结构连接更显著。
- 截至 2026-11-15，官方研究页提供技术全文、PDF 与交互神经元地图；页面**未列出可运行训练/解释 Agent 的独立公开源码仓库入口**。不能把交互可视化当成开源复现代码。
- 文章由研究团队自行发布，文内结果仍需独立复现，不能推广为“所有神经网络都已被解释”。

## 对 wiki 的映射

- [Neurons as an Interpretable Basis 研究详情](../../wiki/entities/paper-neurons-as-interpretable-basis.md)
- [机制可解释性概念](../../wiki/concepts/mechanistic-interpretability.md)
- [深度学习基础](../../wiki/concepts/deep-learning-foundations.md)

## 引用

Tigges, C., Kowal, M., McGrath, T., Lewis, O., & Cammarata, N. (2026). *Neurons as an Interpretable Basis*. Goodfire. <https://www.goodfire.com/research/neurons-as-an-interpretable-basis>
