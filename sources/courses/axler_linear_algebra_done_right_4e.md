# Sheldon Axler · Linear Algebra Done Right（第 4 版）

> 来源归档

- **标题：** Linear Algebra Done Right (LADR), 4th Edition
- **作者：** Sheldon Axler
- **类型：** textbook（PDF 官方免费分发）
- **链接：** <https://linear.axler.net/LADR4e.pdf>
- **配套：** <https://linear.axler.net/>（勘误、习题、Open Access 说明）
- **入库日期：** 2026-05-31（2026-11-07 原文核对）
- **核对版本：** 官方 PDF `LADR4e.pdf`（扉页标注 2026-10-30 版，约 19.6 千行文本）
- **核对范围：** 读完目录（全部 9 章 / 各小节标题）、学生与教师前言（含各章要点概述、「行列式后置」的写作立场）；并在正文中定位核对了与机器人相关的小节：3F Duality、6C 含 Pseudoinverse（前言称可跳过）、7D–7F Matrix Factorization / Singular Value Decomposition / Polar Decomposition / Volume via Singular Values、9C Determinants、9D Tensor Products。**未读：** 各章证明与习题的逐页内容（≈400 页正文未逐页通读），Ch5 特征值专节另见 [`axler_ladr4_ch5_eigenvalues_invariant_subspaces.md`](axler_ladr4_ch5_eigenvalues_invariant_subspaces.md)。本核对只支撑「教材结构与章节主题」的陈述，不支撑「对机器人有何帮助」这类策展判断。
- **通读：** 已完成
- **一句话说明：** 以**向量空间与线性映射**为中心、推迟行列式到后期的经典教材；适合建立「算子 / 不变子空间 / 谱」的严格直觉，再读机器人中的旋转、刚体与 LQR。

## 为什么值得保留

- 本仓库 [运动控制路线 L0](../../roadmap/motion-control.md) 已点名 LADR；此前仅有外链，现归档为可溯源 source。
- **与行列式优先的工科教材不同**：先理解线性映射 \(T:V\to W\)，再落到矩阵表示——与「把 SE(3) 看作群作用」的思维方式一致。
- 第 4 版对复内积空间、多重线性代数等有更新；PDF 与站点公开，便于长期引用。

## 章节与机器人 L0 的映射（不必全书通读）

| LADR 典型章节块 | 机器人相关读法 |
|----------------|----------------|
| 向量空间、线性映射、矩阵表示 | 把 FK、Jacobian 看成线性化后的映射 |
| 不变子空间、特征值、对角化 | 线性化平衡点、模态分析入门 |
| 内积空间、正交投影 | 任务空间投影、WBC 中的最小范数解 |
| 对偶空间、张量（新版加强） | 为阅读 Featherstone / 空间向量文献打底 |
| 行列式（后置章节） | 体积、可逆性；与 ILA 计算课交叉即可 |

## 对 wiki 的映射

- [`wiki/entities/linear-algebra-curriculum.md`](../../wiki/entities/linear-algebra-curriculum.md)
- Ch 5 特征值专节：[axler_ladr4_ch5_eigenvalues_invariant_subspaces.md](axler_ladr4_ch5_eigenvalues_invariant_subspaces.md) → [`wiki/formalizations/eigenvalues-eigenvectors.md`](../../wiki/formalizations/eigenvalues-eigenvectors.md)

## 推荐继续阅读（外部）

- [Interactive Linear Algebra（Georgia Tech）](https://textbooks.math.gatech.edu/ila/)
- [MIT 18.06 Linear Algebra（Gilbert Strang）](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
