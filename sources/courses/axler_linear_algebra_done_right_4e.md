# Sheldon Axler · Linear Algebra Done Right（第 4 版）

> 来源归档

- **标题：** Linear Algebra Done Right (LADR), 4th Edition
- **作者：** Sheldon Axler
- **类型：** textbook（PDF 官方免费分发）
- **链接：** <https://linear.axler.net/LADR4e.pdf>
- **配套：** <https://linear.axler.net/>（勘误、习题、Open Access 说明）
- **入库日期：** 2026-05-31
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

## 推荐继续阅读（外部）

- [Interactive Linear Algebra（Georgia Tech）](https://textbooks.math.gatech.edu/ila/)
- [MIT 18.06 Linear Algebra（Gilbert Strang）](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
