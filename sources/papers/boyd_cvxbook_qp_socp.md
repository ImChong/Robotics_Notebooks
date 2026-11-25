---
type: source-paper
updated: 2026-11-21
url: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
---

# Boyd & Vandenberghe — Convex Optimization（§ 4.4 QP 与 SOCP）

- **作者：** Stephen Boyd（Stanford University）、Lieven Vandenberghe（UCLA）
- **开放全文：** <https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>
- **通读：** 已完成
- **核对范围：** 已通读《Convex Optimization》第 4 章 § 4.4 公开 PDF（Eq. 4.34 标准 QP、Eq. 4.35 QCQP、Eq. 4.36 二阶锥规划 SOCP 及其几何/鲁棒/最小二乘等价形式）。
- **核心形式化：**
  - **QP 标准型（Eq. 4.34）：** `min (1/2) x^T P x + q^T x + r, s.t. Gx <= h, Ax = b`，其中 `P ∈ S_+^n`（半正定对称矩阵保证凸性），可行域为多面体（polyhedron）。
  - **SOCP 标准型（Eq. 4.36）：** `min f^T x, s.t. ||A_i x + b_i||_2 <= c_i^T x + d_i, Fx = g`，为三维库仑摩擦锥（`||f_t||_2 <= μ f_n`）与鲁棒线性约束的自然凸规划表述；当 `A_i = 0` 退化为 LP，多面体切向线性化摩擦锥时 WBC 可降阶为高速 QP。
