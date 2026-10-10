# Genetic Programming：Koza 原著与 ADF 论文

> 来源归档（primary sources）

## 书目与一手入口

- **John R. Koza**, *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press / Bradford Books, 1992：[出版社页](https://mitpress.mit.edu/9780262111706/genetic-programming/)，ISBN 9780262111706。
- **John R. Koza**, “Genetic programming as a means for programming computers by natural selection,” *Statistics and Computing*, 4, 87–112, 1994：[Springer DOI](https://doi.org/10.1007/BF00175355)。
- **作者目录：** [Koza 1992 GP 原著目录](https://www.genetic-programming.com/gpbook1toc.html)，用于章节定位；下列知识归纳以原著和期刊论文为依据。

## 核心摘录与对 Wiki 的映射

1. **搜索对象是程序（Koza 1992）**：在函数集合与终端集合定义的语法范围内生成可执行候选，以任务适应度评估并演化；经典表示为表达式树或 Lisp S-expression。  
   **映射：** [GP 方法页](../../wiki/methods/genetic-programming.md)的个体表示、输入输出与流程。
2. **子树可重组（Koza 1992）**：交叉可交换父代程序中的子树，程序结构和长度随演化改变，表达能力提升但带来树膨胀和计算成本。  
   **映射：** 方法页的子树交叉、复杂度控制与部署边界。
3. **自动定义函数（Koza 1994）**：ADF 允许系统演化可复用子程序，而不局限于单个表达式。  
   **映射：** 方法页的模块化 GP 小节；ADF 不是无代价的代码复用。
4. **执行即评估**：候选本身会运行，适应度设计必须考虑异常、数值发散、约束违反和模拟器失败。  
   **映射：** 方法页的安全沙盒与独立复测。

## 术语边界

本文用“遗传编程（GP）”；它属于进化计算，但不等于用 GA 搜索神经网络权重。前者直接演化受限语法下的程序结构。
