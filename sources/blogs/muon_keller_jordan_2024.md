# Muon: An optimizer for hidden layers in neural networks

> 原始资料归档（ingest）

- **标题：** Muon: An optimizer for hidden layers in neural networks
- **类型：** blog（技术博客，**非 arXiv 论文**）
- **作者：** Keller Jordan、Yuchen Jin、Jeremy Bernstein 等
- **时间：** 2024 年 12 月
- **链接：** https://kellerjordan.github.io/posts/muon/
- **代码：** https://github.com/KellerJordan/Muon
- **入库日期：** 2026-07-14
- **一句话说明：** Muon 的**原始提出**与算法设计动机；GitHub Citation 亦指向本博客而非论文。

## 为什么值得保留

- Muon **并非**从 arXiv 论文首发，而是从 Keller Jordan 博客与开源实现进入社区；理解 Muon 必须从此处读起。
- 定义了 **MomentUm Orthogonalized by Newton-Schulz** 的完整更新规则与 `newtonschulz5` 实现。
- 记录了 NanoGPT speedrun、CIFAR-10 speedrun 等早期实证结果。

## 核心摘录

### 算法定义

Muon 面向神经网络 **隐藏层 2D 参数**（权重矩阵 $W$）：

1. 对梯度做 **SGD-momentum** 得到更新矩阵 $G$；
2. 对 $G$ 施加 **Newton–Schulz 迭代**（`newtonschulz5`，约 5 步）做**近似正交化**；
3. 将正交化后的矩阵作为参数更新方向。

标量/向量参数、输入输出层、以及 4D 卷积（展平后三维）应继续用 **AdamW** 等标准优化器；Muon 只负责隐藏层矩阵块。

### 正交化的含义

Newton–Schulz 迭代近似求解：

$$\mathrm{Ortho}(G) = \arg\min_O \{ \|O - G\|_F : O^\top O = I \text{ 或 } OO^\top = I \}$$

等价于对 $G=USV^\top$ 做 SVD 后取 $UV^\top$，但 **不用完整 SVD**（太慢），而用 bfloat16 友好的迭代。

### 设计动机（博客观点）

- Transformer 隐藏层梯度更新往往 **条件数极高、近似低秩**；正交化可放大「稀有方向」的学习幅度。
- 与 Bernstein & Newhouse (2024) 对 Shampoo 的分析、以及 Shampoo 的谱范数几何有联系。
- 拒绝 SVD（慢）与 coupled Newton iteration（需 float32 才稳定）。

### 早期实证

- CIFAR-10 94% 精度：3.3 → 2.6 A100-秒。
- FineWeb NanoGPT speedrun val loss 3.28：**1.35×** 样本效率。
- 1.5B Transformer 达 GPT-2 XL HellaSwag 水平：Muon **10h** vs AdamW **13.3h**（8×H100）。

## 对 wiki 的映射

- [Muon（方法页）](../../wiki/methods/muon.md)
- [Deep Learning Optimizers 对比](../../wiki/comparisons/deep-learning-optimizers.md)
- [karpathy/autoresearch](../../wiki/entities/karpathy-autoresearch.md)（train.py 默认 Muon+AdamW）
