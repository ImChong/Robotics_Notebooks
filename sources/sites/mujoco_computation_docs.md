# MuJoCo 官方文档：Computation（接触与约束求解）

> 来源归档（官方文档 · 原文通读）

- **标题：** MuJoCo Documentation — Computation（Soft contact model / Constraint model / Constraint solver）
- **类型：** site（官方文档，源文件为 `doc/computation/index.rst`）
- **来源：** Google DeepMind / MuJoCo
- **链接：** <https://mujoco.readthedocs.io/en/stable/computation/index.html>；源文件 <https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/computation/index.rst>
- **入库日期：** 2026-11-07
- **核对版本：** `main` 分支 `doc/computation/index.rst`（2026-11-07 抓取，约 2500 行）
- **核对范围：** 读完 Soft contact model 全节（物理真实性与软接触、计算效率、连续时间）、Constraint model 中 Contact / 摩擦锥小节（elliptic vs pyramidal 定义、接触基向量与雅可比 $E^\top S$）、Constraint solver 的 Primal problem / Algorithms（CG、Newton、PGS、NoSlip、Warmstart）与 Parameters 中阻抗 $d$ 与 $R$ 的关系；**未读：** Actuation model、Numerical integration 的逐积分器细节、Collision detection 全节、Simulation pipeline、Inverse dynamics / Derivatives、Equality / Friction loss / Limit 约束的细节与源码实现。
- **通读：** 已完成

## 关键内容摘录

- **MuJoCo 接触模型是凸模型，不是 LCP：** 放弃严格互补约束，改用凸优化（pyramidal 锥为 QP，elliptic 锥为锥规划）定义约束力。无摩擦时 KKT 条件与 LCP 等价；有摩擦时两者不同。后果：法向力与法向速度可同时为正、摩擦力不一定最大耗散、起滑必伴随一点法向运动——文档称这些效应数值很小，且对软接触反而更真实（软接触必然违反互补）。
- **LCP 的计算代价：** 带摩擦 LCP 对应 NP-hard 问题（最坏情形），导致许多引擎用文档不全的捷径；文档观察到典型机器人模型上 PGS 迭代 10 次与全局最小「实际上无法区分」。
- **连续时间：** 摩擦接触在连续时间下不适定（Painlevé 悖论），所以硬接触引擎常用离散时间速度步进；MuJoCo 的模型直接按连续时间的力与加速度定义（可选 `discrete` 积分器除外）。
- **摩擦锥（condim）：** 椭圆锥 $\mathcal K=\{f: f_1\ge0,\ f_1^2\ge\sum_{i\ge2}f_i^2/\mu_{i-1}^2\}$；棱锥 $\mathcal K=\{f\ge0\}$（基向量是棱锥的边，缩放使其落在被近似的椭圆锥内）。condim 1/3/4/6 分别为无摩擦、切向摩擦、+扭转、+滚动摩擦；棱锥分别用 1/4/6/10 维。接触雅可比为 $E^\top S$。
- **原始问题：** $\dot v=\arg\min_x\|x-M^{-1}(\tau-c)\|_M^2+\|y-a_r\|^{\text{Huber}}_{R^{-1}}$（受约束 $J_{\mathcal C}x-y\in\mathcal K^*$）；$R$ 对角正则使约束变软，$a_r$ 为弹簧–阻尼参考加速度（类 Baumgarte，但改的是优化问题）；阻抗 $d\in(0,1)$ 令 $R_{ii}=\frac{1-d_i}{d_i}\hat A_{ii}$，约束加速度在无约束值与参考值之间插值。
- **求解器：** CG（Hager–Zhang，无初始化成本）；Newton（解析二阶导 + Cholesky，默认）；PGS（对偶形式、一阶收敛，曾是默认；对 elliptic 锥在接触内沿自适应非正交轴更新）；NoSlip 后处理对摩擦维用 $R=0$ 的 PGS 压低打滑，但文档承认它不再对应单一明确的优化问题，复杂多接触下可能不稳；上一步力 warmstart。

## 局限（文档自述）

- 软约束意味着接触有微量穿透 / 滑移，参数（`solref`、`solimp`）需按材料调；NoSlip 是 ad-hoc 修补；$R$ 依赖初始构型 `qpos0` 下的对角近似，在强各向异性惯量或远离 `qpos0` 的长运动链上可能失准。

**对 wiki 的映射：** [`contact-dynamics.md`](../../wiki/concepts/contact-dynamics.md)；仿真实体页 [`mujoco.md`](../../wiki/entities/mujoco.md)
