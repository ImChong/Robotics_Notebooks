# Georgia Tech ILA §5.1 — Eigenvalues and Eigenvectors

> 来源归档（一手教材节选）

- **标题：** Interactive Linear Algebra, Section 5.1 — Eigenvalues and Eigenvectors
- **作者：** Dan Margalit、Joseph Rabinoff（Georgia Institute of Technology）
- **类型：** textbook section
- **链接：** <https://textbooks.math.gatech.edu/ila/eigenvectors.html>
- **入库日期：** 2026-09-11
- **一句话说明：** 方阵特征值/特征向量的标准定义、几何直觉、eigenspace 与可逆矩阵定理扩展；强调 eigen 意为「自身/固有」。

## 核心定义（原文要点）

设 \(A\) 为 \(n\times n\) 矩阵。

1. **特征向量：** 非零向量 \(v\in\mathbb{R}^n\) 满足 \(Av=\lambda v\)（某标量 \(\lambda\)）。
2. **特征值：** 使 \(Av=\lambda v\) 有非平凡解的标量 \(\lambda\)。
3. **术语：** 德语前缀 *eigen* ≈「自身/固有」（characteristic）；零向量**不是**特征向量。
4. **限制：** 仅对方阵定义；特征值可为 0。

## 关键定理与配方

| 结果 | 内容 |
|------|------|
| 几何读法 | \(Av\) 与 \(v\) 共线（过原点同一直线）；\(\lambda\) 为伸缩因子 |
| \(\lambda\)-eigenspace | \(\mathrm{Nul}(A-\lambda I)\) 的子空间；特征向量 = 该零空间非零元 |
| \(\lambda\) 是特征值 \(\Leftrightarrow\) \(A-\lambda I\) 不可逆 |
| 不同特征值的特征向量 | 线性无关 |
| 特征值个数 | \(n\times n\) 矩阵至多有 \(n\) 个特征值 |

## 对 wiki 的映射

- 沉淀 **[`wiki/formalizations/eigenvalues-eigenvectors.md`](../../wiki/formalizations/eigenvalues-eigenvectors.md)**
- 交叉 [`wiki/entities/linear-algebra-curriculum.md`](../../wiki/entities/linear-algebra-curriculum.md)

## 推荐继续阅读（外部）

- [ILA §5.4 Diagonalization](https://textbooks.math.gatech.edu/ila/diagonalization.html)
- [Axler LADR Ch 5](https://linear.axler.net/LADR4e.pdf)
