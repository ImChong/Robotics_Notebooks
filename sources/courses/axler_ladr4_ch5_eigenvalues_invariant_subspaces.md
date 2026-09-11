# Axler LADR4e Ch 5 — Eigenvalues, Eigenvectors, and Invariant Subspaces

> 来源归档（一手教材节选）

- **标题：** Linear Algebra Done Right, 4th Edition — Chapter 5
- **作者：** Sheldon Axler
- **类型：** textbook chapter
- **链接：** <https://linear.axler.net/LADR4e.pdf>（Ch 5, §5A–5B）
- **许可：** Open Access PDF
- **入库日期：** 2026-09-11
- **一句话说明：** 从**不变子空间**出发定义特征值：一维不变子空间 \(\Leftrightarrow\) 存在 \(\lambda\) 使 \(Tv=\lambda v\)；强调算子视角而非行列式优先。

## 核心定义（§5A，原文要点）

| 概念 | 定义 |
|------|------|
| 不变子空间 | \(U\subset V\) 且 \(T(U)\subset U\) |
| 特征值 | \(T\in\mathcal{L}(V)\) 的 \(\lambda\in\mathbb{F}\) 使得存在 \(v\neq 0\) 且 \(Tv=\lambda v\) |
| 特征向量 | 对应特征值 \(\lambda\) 的非零 \(v\) 满足 \(Tv=\lambda v\) |
| 术语 | *eigen* = German「自身/固有」；排除 \(v=0\) 因为 \(T0=0\) 对一切 \(\lambda\) 成立 |

## 等价条件（有限维，定理 5.7）

对 \(T\in\mathcal{L}(V)\)、\(\lambda\in\mathbb{F}\)，以下等价：

- (a) \(\lambda\) 是 \(T\) 的特征值
- (b) \(T-\lambda I\) 非单射
- (c) \(T-\lambda I\) 非满射
- (d) \(T-\lambda I\) 不可逆

**推论：** \(\lambda\) 是特征值 \(\Leftrightarrow\) \(\lambda\in\mathrm{Nul}(T-\lambda I)\) 有非零向量。

## 关键定理

- **5.11：** 对应**不同**特征值的特征向量列表线性无关
- **5.12：** 算子特征值个数 \(\le \dim V\)
- **5B：** 复向量空间上算子必有特征值（代数基本定理）；实空间奇数维必有实特征值

## 与行列式路线的关系

Axler 将行列式推迟到后期章节；特征值先由**一维不变子空间**定义，与机器人中「模态方向不变、仅缩放」的直觉一致。

## 对 wiki 的映射

- 沉淀 **[`wiki/formalizations/eigenvalues-eigenvectors.md`](../../wiki/formalizations/eigenvalues-eigenvectors.md)**
