# Genetic Algorithms：Holland 与后续奠基资料

> 来源归档（primary sources）

## 书目与一手入口

- **John H. Holland**, *Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence*. 初版 University of Michigan Press, 1975；MIT Press 1992 修订版：[出版社页](https://mitpress.mit.edu/9780262082136/adaptation-in-natural-and-artificial-systems/)，[DOI](https://doi.org/10.7551/mitpress/1090.001.0001)。
- **John H. Holland**, “Outline for a Logical Theory of Adaptive Systems,” *Journal of the ACM*, 9(3), 297–314, 1962：[ACM DOI](https://dl.acm.org/doi/10.1145/321127.321128)。
- **David E. Goldberg & John H. Holland**, “Genetic Algorithms and Machine Learning,” *Machine Learning*, 3, 95–99, 1988：[Springer DOI](https://doi.org/10.1007/BF00113892)。
- **David E. Goldberg**, *Genetic Algorithms in Search, Optimization, and Machine Learning*, Addison-Wesley, 1989：[书目记录](https://books.google.com/books?id=2IIJAAAACAAJ)。

## 核心摘录与对 Wiki 的映射

1. **适应与遗传搜索（Holland 1975）**：把自然选择、编码结构、适应度、重组与 schemata 放进统一的自适应系统框架。  
   **映射：** [GA 方法页](../../wiki/methods/genetic-algorithm.md)的编码、选择和适应度；模式定理不是任意 GA 都能快速找到全局最优的保证。
2. **逻辑自适应系统（Holland 1962）**：提供早期 classifier systems 与遗传搜索理论的原始背景。  
   **映射：** [强化学习史](../../wiki/concepts/reinforcement-learning-history.md)中进化搜索与 RL 的边界。
3. **GA 与机器学习（Goldberg & Holland 1988）**：短篇领域导言梳理遗传算法与机器学习的交界，并引用 Holland 的早期工作。  
   **映射：** GA 虽可按 fitness 选择，但不是因而自动成为 RL；两者的信息结构和学习目标不同。
4. **优化实践（Goldberg 1989）**：系统讨论编码、遗传算子、选择及搜索/优化应用。  
   **映射：** 方法页的算子取舍与机器人参数优化步骤。

## 机器人示例

仓库已有 [GA 坡面双足](../../wiki/entities/paper-ga-biped-slope-gait.md)：在 8-DoF 双足模型上搜索低维步态轨迹参数并以 ZMP 惩罚筛选。它是离线轨迹参数优化案例，不等价于直接用 GA 学高频闭环策略。
