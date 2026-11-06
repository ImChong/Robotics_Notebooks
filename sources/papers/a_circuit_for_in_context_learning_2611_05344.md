# A Circuit for In-Context Learning / 上下文学习的电路

- **作者及机构：** Andrew Gao、William Gao，斯坦福大学（Stanford University）
- **版本：** [arXiv:2611.05344v1](https://arxiv.org/abs/2611.05344v1)，2026-11-03
- **原文：** [HTML](https://arxiv.org/html/2611.05344v1) · [PDF](https://arxiv.org/pdf/2611.05344v1)
- **代码：** [官方仓库](https://github.com/andrewgcodes/in-context-learning-circuit)
- **入库日期：** 2026-11-06

## 研究摘要

以短句、任意标签的少样本分类为对象，通过激活替换、头消融及方向干预研究示例写入、门控和检索。原文 §14 明确限制为短句、单 token 标签与最多 12 个示例的主要设置，并未解释长文本、指令遵循或多类绑定。

- §3–5：类别写入标签位置，重复检测信号调节读取；不同类中某一标签仅出现一次仍可读取。
- §7–9：读取头和抑制近期偏置的头按群体作用；三个基础模型的具体实现存在差异。
- §10–12：测试自然概念、有语义标签、不同大小及指令微调；手工构造和训练玩具模型。
- §14：12 示例上拟合的读取规律不能外推到 18 示例；部分预测失败。
- **核对进度：** 已核读摘要、方法、主要实验和讨论，附录及全部源码尚未逐段核读；此归档为草稿。

## 对 wiki 的映射

- [上下文学习的电路](../../wiki/entities/paper-a-circuit-for-in-context-learning.md)
