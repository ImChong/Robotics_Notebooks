# MIT 6.007 — Lecture 21: Continuous-Time Second-Order Systems

> 来源归档（一手课程讲义）

- **标题：** RES.6-007 Signals and Systems — Lecture 21: Continuous-Time Second-Order Systems
- **作者：** Alan V. Oppenheim（MIT）
- **类型：** lecture notes (PDF)
- **链接：** <https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/10c777db2cf27ca7a55b08d3fd6e3e78_MITRES_6_007S11_lec21.pdf>
- **入库日期：** 2026-09-11
- **一句话说明：** 用拉普拉斯变换与系统函数 $H(s)$ 统一描述一阶（单极点）与二阶（极点对）系统；欠阻尼对应复共轭极点与时域衰减振荡。

## 核心要点（原文摘录）

### 系统函数与极点

- 线性常系数微分方程 → **系统函数** $H(s)$；因果系统 ROC 为最右极点右侧。
- **一阶系统：** $s$ 平面单极点。
- **二阶系统：** 一对极点；两实极点 → **过阻尼（overdamped）**；复共轭对 → **欠阻尼（underdamped）**，冲激响应为带指数包络的振荡。

### 与微分方程的对应

对二阶系统，由系数关系可判定极点在实轴或成共轭复对；**欠阻尼**时域响应为振荡并被指数衰减调制。

### 工程读法

一阶、二阶系统是更高阶系统的**积木**；多极点响应可分解为各阶模态叠加。

## 对 wiki 的映射

- 沉淀 **[`wiki/formalizations/damped-systems.md`](../../wiki/formalizations/damped-systems.md)** — $s$ 平面极点与 $\zeta$ 分类
- 交叉 [特征值与特征向量](../../wiki/formalizations/eigenvalues-eigenvectors.md)（极点 = 特征值）

## 推荐继续阅读（外部）

- [MIT 6.007 OCW 课程站](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/)
- [Lecture 20: Laplace Transform](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/resources/lecture-20-the-laplace-transform/)
