# Gilbert Strang · Introduction to Linear Algebra 5e Ch 6 + MIT 18.06 Lecture 21

> 来源归档（一手教材 + 课程）

- **标题：** Eigenvalues and Eigenvectors（ILA 5th ed. Ch 6；MIT 18.06 Lecture 21）
- **作者：** Gilbert Strang
- **类型：** textbook chapter + lecture
- **教材：** *Introduction to Linear Algebra*, 5th ed., Wellesley-Cambridge Press, 2016, ISBN 9780980232776
- **教材站：** <https://math.mit.edu/~gs/linearalgebra/ila5/index.html>
- **目录 PDF：** <https://math.mit.edu/~gs/linearalgebra/ila5/linearalgebra5_TOC.pdf>（Ch 6 p.288）
- **课程：** <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-21-eigenvalues-and-eigenvectors/>
- **入库日期：** 2026-09-11
- **一句话说明：** Strang 四大子空间叙事下的特征值章：\(Ax\) 与 \(x\) 同向 \(\Rightarrow\) \(x\) 为特征向量；Ch 6.1–6.5 覆盖对角化、微分方程、对称与正定矩阵。

## Ch 6 结构（目录）

| 节 | 主题 |
|----|------|
| 6.1 | Introduction to Eigenvalues |
| 6.2 | Diagonalizing a Matrix |
| 6.3 | Systems of Differential Equations |
| 6.4 | Symmetric Matrices |
| 6.5 | Positive Definite Matrices |

## Lecture 21 核心表述（OCW）

> If the product \(Ax\) points in the same direction as the vector \(x\), we say that \(x\) is an **eigenvector** of \(A\). Eigenvalues and eigenvectors describe what happens when a matrix is multiplied by a vector.

## 工程读法（Strang 传统）

- **特征值 \(\Leftrightarrow\) 主方向上的伸缩率** — 与机器人线性化系统 \( \dot x = Ax \) 的模态直接对应
- **对称矩阵实特征值** — 惯量张量、协方差矩阵、Hessian 的谱分析基础
- **正定矩阵** — LQR 中 \(Q,R\) 正定、能量函数凸性

## 对 wiki 的映射

- 沉淀 **[`wiki/formalizations/eigenvalues-eigenvectors.md`](../../wiki/formalizations/eigenvalues-eigenvectors.md)**
- 交叉 [`wiki/formalizations/lqr.md`](../../wiki/formalizations/lqr.md)
