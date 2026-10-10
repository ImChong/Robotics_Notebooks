# 连续黑箱优化的进化算法原始论文

> 来源归档（primary sources）

## 书目与一手入口

- **Nikolaus Hansen & Andreas Ostermeier**, “Completely Derandomized Self-Adaptation in Evolution Strategies,” *Evolutionary Computation*, 9(2), 159–195, 2001：[MIT Press DOI](https://doi.org/10.1162/106365601750190398)，[PubMed 摘要](https://pubmed.ncbi.nlm.nih.gov/11382355/)。
- **Rainer Storn & Kenneth Price**, “Differential Evolution—A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces,” *Journal of Global Optimization*, 11, 341–359, 1997：[Springer DOI](https://doi.org/10.1023/A:1008202821328)。

## 核心摘录与对 Wiki 的映射

1. **协方差矩阵自适应（Hansen & Ostermeier 2001）**：CMA-ES 适应高斯变异分布的协方差和步长，使连续变量搜索调整尺度及变量相关方向。  
   **映射：** [GA 方法页](../../wiki/methods/genetic-algorithm.md)的连续优化选型；连续参数不必先二值编码。
2. **差分进化（Storn & Price 1997）**：从种群向量差分生成试验候选，经交叉与适应度择优；原论文关注连续、可能非线性且不可微目标。  
   **映射：** 方法页的连续黑箱对照：DE 可直接作用于实数向量，但仍需要稳定的目标评估。
3. **工程判断**：这些路线的关键差异在解表示、变异方向/分布、选择规则和评估预算，不是哪种方法“更像生物”。  
   **映射：** GA 方法页的机器人选型与仿真预算建议。
