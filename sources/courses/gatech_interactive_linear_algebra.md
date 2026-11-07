# Georgia Tech · Interactive Linear Algebra（ILA）

> 来源归档

- **标题：** Interactive Linear Algebra
- **作者：** Dan Margalit、Joseph Rabinoff（Georgia Institute of Technology, School of Mathematics）
- **类型：** course / textbook（交互式在线教材）
- **链接：** <https://textbooks.math.gatech.edu/ila/>
- **许可：** 开源教材站点（页面支持 GitHub 提交勘误）
- **入库日期：** 2026-05-31（2026-11-07 尝试核对）
- **核对范围：** 2026-11-07 尝试抓取 <https://textbooks.math.gatech.edu/ila/>：HTTPS 证书链校验失败（未关闭校验），WebFetch 返回 503，源码仓库 `QBobWatson/ila` 仅 README / DEVELOP.md 可经 raw 取得，**未能读到任何章节正文**；本归档下方「章节与机器人 L0 的映射」仍是 2026-05-31 的策展整理，未经原文核实。
- **通读：** 未完成
- **一句话说明：** 面向本科的**交互式**线性代数教材：强调几何直觉、行化简与列空间/零空间，配套可点击例题与可视化，适合与机器人 SE(3) 前的矩阵语言打底并行阅读。

## 为什么值得保留

- **免费、可在线阅读**，章节结构清晰，比纯 PDF 更易做「读一节、做一节」的 L0 打底。
- **几何与代数并重**：对理解旋转矩阵、投影、最小二乘（IK / 状态估计）有直接帮助。
- 与 [Axler LADR](axler_linear_algebra_done_right_4e.md) 形成互补：ILA 更偏计算与可视化直觉，LADR 更偏向量空间公理化。

## 章节与机器人 L0 的映射（精读建议）

| ILA 主题（站点目录） | 机器人里何时用到 |
|---------------------|------------------|
| 向量、矩阵乘法、线性变换 | 关节空间 ↔ 任务空间、齐次变换复合 |
| 行列式、可逆性 | 雅可比奇异性、约束满秩判断 |
| 子空间、列空间 / 零空间、秩 | 冗余机械臂 IK、可观测性直觉 |
| 正交性、最小二乘、QR | 数值 IK、状态估计、Moore–Penrose 伪逆 |
| 特征值 / 特征向量、对角化 | 线性系统稳定性、LQR 中 \(A\) 的谱 |
| 对称矩阵、SVD（若课程含） | 数据驱动降维、病态最小二乘 |

## 对 wiki 的映射

- [`wiki/entities/linear-algebra-curriculum.md`](../../wiki/entities/linear-algebra-curriculum.md)
- 特征值专节归档：[§5.1 Eigenvalues and Eigenvectors](gatech_ila_sec5_1_eigenvalues_eigenvectors.md) → [`wiki/formalizations/eigenvalues-eigenvectors.md`](../../wiki/formalizations/eigenvalues-eigenvectors.md)

## 推荐继续阅读（外部）

- [Linear Algebra Done Right 4e（Axler）](https://linear.axler.net/LADR4e.pdf)
- [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
