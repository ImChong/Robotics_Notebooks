# Boyd & Vandenberghe: Convex Optimization（教材通读归档）

> 来源归档

- **标题：** *Convex Optimization*
- **作者：** Stephen Boyd（Stanford University）、Lieven Vandenberghe（UCLA）
- **类型：** textbook
- **出版信息：** Cambridge University Press, First published 2004, Seventh printing with corrections 2009（ISBN 978-0-521-83378-3；官方免费电子书：<https://web.stanford.edu/~boyd/cvxbook/>）
- **入库日期：** 2026-11-19
- **核对范围：** 2026-11-19 通读 `bv_cvxbook.txt`（Cambridge University Press 2004 / 7th printing 2009 文本版）的 Preface、完整目录，第 I–II 部分全部理论与标准问题章节（Ch 1 导论与复杂度分级、Ch 2 凸集与对偶锥、Ch 3 凸函数与共轭函数、Ch 4 凸优化标准型与锥规划、Ch 5 拉格朗日对偶与 KKT 最优性条件），以及第 III 部分全部算法章节（Ch 9 无约束优化与自和谐理论、Ch 10 等式约束与不可行起始牛顿法、Ch 11 内点法：对数障碍函数与原始–对偶内点法）；未通读第 6–8 章（逼近、统计估计与几何问题应用章节）及附录细节。
- **通读：** 已完成
- **一句话说明：** 凸优化领域最具影响力的标准研究生教材，建立「凸建模识别—对偶与 KKT 理论—牛顿与内点法多项式复杂度及稀疏/块消元实现」完整体系，是理解机器人控制中 LP/QP/SOCP/SDP 与最优控制稀疏 KKT 求解的理论基石。
- **沉淀到 wiki：** 是 → [`wiki/entities/numerical-optimization-curriculum.md`](../../wiki/entities/numerical-optimization-curriculum.md)

## 核心摘录（逐章通读整理）

### 1) 第 1 章导论：问题分级、计算复杂度与实时控制背景

- **三类基础范式对比（§1.1–1.4）：**
  - **最小二乘（LS）：** 解析解可经 QR/正规方程求解，约 $O(n^2 k)$ 复杂度（$k \ge n$）。
  - **线性规划（LP）：** 无解析解但内点法可在多项式时间可靠求解，典型求解代价约 $O(n^2 m)$（$m \ge n$）。
  - **一般凸优化：** 局部最优即全局最优；无通用闭式解，但内点法通常在 10–100 步内迭代收敛至高精度，单步代价量级约 $\max\{n^3, n^2 m, F\}$（$F$ 为函数及二阶导求值开销）。
- **机器人与在线控制线索：** Ch 1 文献注释明确提及将凸优化内嵌于低延迟实时控制与滚动时域模型预测控制（MPC）的工程方向；与一般局部/全局非凸优化（无最优性保证或指数级最坏复杂度）形成本质边界。

### 2) 第 2–3 章：凸集、广义不等式与凸函数

- **几何与锥体系（Ch 2）：** 定义仿射集、凸集、多面体（polyhedra）、二阶锥 / Lorentz 锥、半正定锥 $\mathbb{S}^n_+$；讨论保凸运算（交、仿射变换、透视与线性分式变换）与线性矩阵不等式（LMI）。引入恰当锥（proper cone）诱导的广义不等式 $\preceq_K$、分离/支撑超平面定理及对偶锥 $K^*$（非负象限、SOC 与 $\mathbb{S}^n_+$ 均为自对偶锥）。
- **函数判定与变换（Ch 3）：** 建立凸函数的一阶条件（$f(y) \ge f(x) + \nabla f(x)^T(y-x)$）、二阶条件（$\nabla^2 f(x) \succeq 0$）、上境图（epigraph）与 Jensen 不等式、Schur 补与透视保凸规则；系统定义 Fenchel 共轭 $f^*(y) = \sup_{x \in \mathrm{dom} f}(y^T x - f(x))$、拟凸函数（quasiconvexity）、对数凹凸性以及相对恰当锥的 $K$-凸性。

### 3) 第 4 章：凸优化问题谱系与等价变换

- **标准型与全局性（§4.1–4.2）：** 标准凸优化问题要求目标函数 $f_0$ 与不等式约束 $f_1,\dots,f_m$ 为凸函数、等式约束为仿射形式 $Ax=b$。在该定义下，**任意局部最优解必为全局最优解**；对于可微 $f_0$，点 $x$ 最优当且仅当 $\nabla f_0(x)^T(y-x) \ge 0$ 对所有可行点 $y$ 成立（支撑超平面几何解释）。
- **凸问题层次（§4.3–4.6）：** 严格梳理从简单到通用的包含关系：
  - **LP $\subseteq$ QP $\subseteq$ QCQP $\subseteq$ SOCP $\subseteq$ SDP**：涵盖线性目标加正交多面体/椭球约束，以及含不确定椭球或概率机会约束的鲁棒 LP（§4.4.2 化为 SOCP）和含矩阵线性不等式 LMI 的半正定规划（§4.6.2）。
  - **几何规划（GP，§4.5）：** 正项式/单项式形式虽非天然凸，但通过取对数变量替换 $y_i = \log x_i$ 与对数和指数（log-sum-exp）变换等价化为标准凸问题。
  - **拟凸与向量/多目标优化（§4.2.5, §4.7）：** 拟凸问题可存在非全局的局部最优，可用二分法配凸可行性问题求解；向量优化定义 Pareto 最优解与标量化（$\lambda^T f_0(x)$）权衡曲面。

### 4) 第 5 章：拉格朗日对偶、Slater 条件与 KKT 力学解释

- **弱对偶与强对偶（§5.1–5.3）：** 拉格朗日对偶函数 $g(\lambda, \nu) = \inf_{x \in D} L(x,\lambda,\nu)$ 恒为凹函数（即便原问题非凸），弱对偶 $d^\star \le p^\star$ 无条件成立。对凸问题而言，若存在严格可行点（**Slater 条件**，含针对仿射不等式的精化 Slater 条件），则强对偶 $d^\star = p^\star$ 成立且对偶最优达到。
- **KKT 条件与停机准则（§5.5）：** 对任何满足强对偶的可微问题，KKT（原始可行、对偶可行 $\lambda^\star \succeq 0$、互补松弛 $\lambda_i^\star f_i(x^\star) = 0$、平稳性 $\nabla_x L = 0$）为**必要条件**；**若原问题为凸且可微，KKT 条件本身即为充要条件**（满足 Slater 时则等价）。原始–对偶可行对提供的对偶间隙 $f_0(x) - g(\lambda,\nu)$ 给出非启发式停机认证。§5.5.4 用弹簧–物块力学平衡直观解释：KKT 乘子正是接触力（单向非负、仅在接触激活时非零）。
- **灵敏度与择一性定理（§5.6, §5.8）：** 局部最优值对约束微扰的偏导等于负的最优乘子（影子价格解释）；Farkas 引理及凸不等式系统择一性定理为不可行性认证（Phase I）提供对偶依据。

### 5) 第 9–11 章：无约束、等式约束、内点法与控制结构加速

- **无约束下降与牛顿法（Ch 9）：** 剖析回溯线搜索（backtracking $\alpha \in (0, 0.5), \beta \in (0,1)$）下的梯度下降与最速下降法，指出一阶线性收敛率强依赖子水平集/Hessian 条件数 $\kappa = M/m$；对比**牛顿法**的仿射不变性（affine invariance）、牛顿减量 $\lambda(x) = (\nabla f^T \nabla^2 f^{-1} \nabla f)^{1/2}$ 以及阻尼阶段 + 局部二次收敛阶段（仅需 ~5–6 步达到机器精度）。§9.6 阐述 Nesterov–Nemirovski **自和谐条件**（self-concordance：$|f'''(x)| \le 2 f''(x)^{3/2}$），给出不依赖未知常数 $m,M,L$ 的显式迭代复杂度上界 $(f(x^{(0)})-p^\star)/\gamma + 6$。§9.7 讨论带状 $O(nk^2)$、稀疏 Cholesky 与对角加低秩结构的加速。
- **等式约束与最优化控制结构（Ch 10）：** 将等式约束二次逼近归结为对称不定线性 KKT 系统（$\begin{bmatrix}\nabla^2 f & A^T \\ A & 0\end{bmatrix}$），证明其与零空间消元完全等价；给出**不可行起始牛顿法**（Algorithm 10.2），以残差范数 $\|r(x,\nu)\|_2$ 做线搜索并在达到满步长 $t=1$ 后自动保持原始可行。§10.4.3 **最优控制算例（p. 552–553）**关键演示：将有限时域线性动力学 $z(t+1)=A_t z(t)+B_t u(t)$ 显式写作带块对角 Hessian 的等式约束问题后，KKT Schur 补 $S = -A H^{-1} A^T$ 呈**块三对角**结构，使得单步牛顿系统求解代价从通用稠密的 $O(N^3 (2k+l)^3)$ 降至与预测视界 $N$ 成线性的 **$O(k^3 N)$**（等价于经典 Riccati 递推）。
- **内点法与路径跟踪（Ch 11）：**
  - **对数障碍与中心路径（§11.2–11.5）：** 用 $\phi(x) = -\sum_{i=1}^m \log(-f_i(x))$ 光滑逼近指示函数；中心路径点 $x^\star(t)$ 对应将互补松弛变形为 $-\lambda_i f_i(x) = 1/t$，自然诱导对偶可行点且对偶间隙严格等于 $m/t$。障碍法（Algorithm 11.1）通过外层倍增 $t := \mu t$（工程推荐 $\mu \in [10,100]$）、内层牛顿定心求解；基于自和谐理论，取 $\mu = 1 + 1/\sqrt{m}$ 可证明最坏总牛顿步数上界为 $O(\sqrt{m})$（实践中常为几十步且几乎不随维度 $m,n$ 增长）。§11.4 系统给出 Phase I 可行性问题（最小化最大违背 $s$ 或违背之和）以及与不可行起始牛顿法的对比。
  - **广义不等式与原始–对偶内点法（§11.6–11.7）：** 借助恰当锥的广义对数障碍（SOC 次数 $\theta_i=2$，PSD 锥 $\log\det X$ 次数 $p$），中心路径对偶间隙推广为 $\theta/t$；**原始–对偶内点法**（Algorithm 11.2）直接对未消元修改 KKT 残差 $r_t(x,\lambda,\nu)=0$ 走单循环牛顿步，定义代理对偶间隙 $\hat{\eta}(x,\lambda) = -f(x)^T \lambda$ 并采用最大允许步长的 0.99 倍截断（line search 保证 $f(x)\prec 0, \lambda\succ 0$），在高精度与实测迭代数上通常优于经典双层障碍法。

## 对 wiki 的映射

- [`wiki/entities/numerical-optimization-curriculum.md`](../../wiki/entities/numerical-optimization-curriculum.md) — 核心理论策展页（作为 `original:` 一手教材底座）
- [`wiki/formalizations/convex-functions.md`](../../wiki/formalizations/convex-functions.md) · [`wiki/formalizations/kkt-conditions.md`](../../wiki/formalizations/kkt-conditions.md)
- [`wiki/formalizations/quadratic-programming.md`](../../wiki/formalizations/quadratic-programming.md) · [`wiki/formalizations/symmetric-cone-programming.md`](../../wiki/formalizations/symmetric-cone-programming.md)
- [`wiki/methods/line-search-steepest-descent.md`](../../wiki/methods/line-search-steepest-descent.md) · [`wiki/methods/newtons-method.md`](../../wiki/methods/newtons-method.md) · [`wiki/methods/penalty-barrier-augmented-lagrangian.md`](../../wiki/methods/penalty-barrier-augmented-lagrangian.md)

## 关联资料

- [`sources/papers/second_order_optimizers.md`](../papers/second_order_optimizers.md) — 牛顿/拟牛顿/LM 等一阶与二阶优化器出处汇编
- [`sources/courses/numerical_optimization_foundations_robotics.md`](numerical_optimization_foundations_robotics.md) — 中文课程六章大纲（保留独立对照，标注为`未完成`）
