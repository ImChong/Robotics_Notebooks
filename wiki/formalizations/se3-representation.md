---
type: formalization
tags: [kinematics, math, deep-learning, rotation]
status: complete
updated: 2026-11-07
related:
  - ../overview/modern-robotics-wechat-principles-series.md
  - ./homogeneous-coordinates-transform.md
  - ./lie-group-rigid-body-motions.md
  - ./unit-quaternion-so3.md
  - ./tan-norm-rotation.md
  - ../comparisons/so3-rotation-representations.md
  - ../concepts/whole-body-control.md
  - ../methods/visual-servoing.md
  - ../formalizations/mdp.md
  - ../entities/paper-se3-tangent-to.md
  - ../entities/mimickit.md
  - ../methods/trajectory-optimization.md
sources:
  - ../../sources/blogs/wechat_goodman_modern_robotics_ch3_rotation_angular_velocity.md
  - ../../sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md
  - ../../sources/papers/perception.md
  - ../../sources/papers/se3_tangent_to_arxiv_2508_11520.md
  - ../../sources/papers/diebel_2006_representing_attitude_quaternions.md
  - ../../sources/papers/shoemake_1985_quaternion_curves_siggraph.md
  - ../../sources/papers/zhou_2019_cvpr_continuity_rotation_representations.md
  - ../../sources/repos/mimickit_tan_norm.md
original:
  - ../../sources/papers/modern_robotics_textbook.md
  - ../../sources/papers/zhou_2019_cvpr_continuity_rotation_representations.md
  - ../../sources/papers/diebel_2006_representing_attitude_quaternions.md
summary: "SE(3) 位姿表示形式化：探讨了欧拉角、四元数、旋转矩阵及 6D 连续表示在机器人学习中的优劣对比，重点关注其在神经网络训练中的连续性与独特性。"
---

# SE(3) Representation (位姿表示形式化)

## 一句话定义

在机器人学与具身智能中，如何表示物体的**位姿（Pose）**——即位置与姿态的组合，是感知与控制的基础。**SE(3)** (Special Euclidean Group) 描述了三维空间中的刚体运动。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SE(3) | Special Euclidean Group in 3D | 三维刚体位姿（旋转 + 平移）所在的群，用 $4\times4$ 齐次矩阵表示 |
| SO(3) | Special Orthogonal Group in 3D | 三维旋转群：$R^\top R=I$，$\det R=1$ |
| 6D | 6D continuous rotation representation | Zhou et al. 2019：取 $R$ 前两列，经 Gram–Schmidt 还原第三列 |
| SLERP | Spherical Linear Interpolation | 单位四元数球面上的插值 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |

## 为什么重要

位姿估计、末端控制、轨迹优化、VLA 动作头都要在「怎么写一个姿态」上做选择：同一个旋转可以写成 9 个数（矩阵）、4 个数（四元数）、3 个数（欧拉角 / 指数坐标）或 6 个数（连续表示），它们在奇异、约束和神经网络可学性上差别很大（Zhou et al. 2019 证明：3D 旋转在 4 维及以下的实欧氏空间中没有任何连续表示）。选错表示，常见后果是万向锁、$q\equiv -q$ 的损失跳变、插值后不再是刚体旋转。

## 核心原理（数学定义）

一个 SE(3) 元素通常由两部分组成：
- **位置 (Translation)**：$t \in \mathbb{R}^3$。
- **姿态 (Rotation)**：属于 SO(3) 组，$R \in SO(3)$。

$$ T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4} $$

*Modern Robotics* §3.2–3.3 给出的结构性质：$R^{-1}=R^\top$，$T^{-1}=\begin{bmatrix}R^\top & -R^\top p\\0&1\end{bmatrix}$；群乘法 $T_{ab}T_{bc}=T_{ac}$（下标消去规则）、$T_{ab}p_b=p_a$；乘法对 SO(3) 一般**不可交换**。绕固定系轴转用左乘 $RR_{sb}$，绕体坐标轴转用右乘 $R_{sb}R$。角速度满足 $\dot RR^{-1}=[\omega_s]$、$R^{-1}\dot R=[\omega_b]$。任意旋转有指数坐标 $\hat\omega\theta$：$R=e^{[\hat\omega]\theta}=I+\sin\theta[\hat\omega]+(1-\cos\theta)[\hat\omega]^2$（Rodrigues），反过来由 $\mathrm{tr}R=1+2\cos\theta$ 与 $[\hat\omega]=\frac1{2\sin\theta}(R-R^\top)$ 求对数，$\theta=\pi$（$\mathrm{tr}R=-1$）时轴需另行求解；SO(3) 可视为半径 $\pi$ 的实心球（对径点等同）。位姿则有 $T=e^{[\mathcal S]\theta}$，$\mathcal S=(\omega,v)\in\mathbb{R}^6$ 为螺旋轴，换系用伴随 $[\mathrm{Ad}_T]=\begin{bmatrix}R&0\\ [p]R&R\end{bmatrix}$。

SO(3) 七种表示的维度、奇异、插值与选型决策见专页 [旋转表示方法对比](../comparisons/so3-rotation-representations.md)；本页只保留位姿 $T=(R,t)$ 与 **面向神经网络** 的短表。

## 主流表示法对比（面向神经网络）

在深度学习模型（如 VLA 或 Pose Estimation）中，选择姿态表示法至关重要，因为它直接影响梯度的平滑性和损失函数的收敛。

| 表示法 | 维度 | 优势 | 劣势 | 推荐场景 |
|------|-----|-----|-----|---------|
| **欧拉角 (Euler Angles)** | 3 | 直观，最省空间 | 存在万向节死锁 (Gimbal Lock)；不连续 | 简单的 UI 显示 |
| **四元数 (Quaternions)** | 4 | 紧凑，无死锁，插值平滑 | 存在双倍覆盖 ($q = -q$)；单位化约束 | 控制器内部状态、MoCap |
| **旋转矩阵 (Rotation Matrix)** | 9 | 线性，无死锁，唯一 | 自由度冗余 (9D 表示 3D)；需要正交化 | 坐标变换计算 |
| **6D 连续表示 (6D Rep)** | 6 | **姿态空间连续**；适合神经网络回归 | 需要格拉姆-施密特正交化 | **Deep Learning 姿态估计** |
| **tan_norm** | 6 | 连续；无 $q \equiv -q$；体 x/z 轴语义 | 命名非文献通用；参考轴固定 | **MimicKit / ProtoMotions 运动模仿观测** |

连续性的理论依据（Zhou et al., CVPR 2019）：定义「表示 $g:SO(3)\to\mathbb{R}^n$ 连续」后证明，欧拉角（恒等旋转处 $0$ 与 $2\pi$ 两侧极限不同）、四元数（180° 旋转集附近映射不连续）、轴角都不连续；$SO(3)\cong\mathbb{RP}^3$ 只能嵌入 $\mathbb{R}^5$ 及以上，所以 $\le4$ 维不存在连续表示，6D（$n^2-n$ 维 Gram–Schmidt）与 5D（再做立体投影）连续。实验（自编码器、点云姿态、人体 IK）中 6D / 5D 的平均测地误差显著小于四元数 / 轴角 / 欧拉，但作者也发现降维的 5D 并不优于 6D。

Hamilton 积、SLERP 与 **scalar 顺序** 见 [单位四元数与 SO(3)](./unit-quaternion-so3.md)。

### 1. 为什么 6D 表示法更适合 DL？
传统的四元数和欧拉角在 $\mathbb{R}^n$ 到 $SO(3)$ 的映射过程中存在**不连续点**。这意味着当网络预测值发生微小连续变化时，映射出的旋转可能发生突变。
6D 表示法通过取旋转矩阵的前两列 $(a_1, a_2)$，并利用正交化重建第三列，实现了全空间的连续映射。

运动模仿栈里的 **[tan_norm](./tan-norm-rotation.md)** 同属 6 维连续族，但编码为 $[\,R(q)\mathbf{t}_0 \,\|\, R(q)\mathbf{n}_0\,]$（固定参考切向/法向），与 Zhou 论文「任意前两列」在实现与语义上不同——读 MimicKit 观测维数时必须区分二者。

## 损失函数 (Loss Functions)

对于 SE(3) 位姿回归，常见的损失函数包括：
- **L2 距离**：分别计算位置和姿态分量的均方误差。
- **测地线距离 (Geodesic Distance)**：姿态误差最真实的物理描述：
  $$ \mathcal{L}_{rot} = \arccos\left( \frac{\text{Tr}(R_{pred} R_{target}^T) - 1}{2} \right) $$

**测地损失的由来：** 两旋转之间的相对角 $\theta$ 满足 $\mathrm{tr}(R_{pred}R_{target}^\top)=1+2\cos\theta$（*Modern Robotics* 式 3.54），所以 $\arccos\frac{\mathrm{tr}(\cdot)-1}{2}$ 就是旋转角；Zhou et al. 实验也以此作为评测误差。

## 工程实践

- **存储与复合：** 状态用 $T=(R,t)$ 或 $(q,t)$；链式复合用矩阵乘；四元数每次积分后重新归一化（Diebel §1：单位范数是二次约束，多数标准优化算法无法直接编码）。
- **优化变量：** 对姿态优化，在 $\mathfrak{so}(3)$ 上取增量 $R\leftarrow R\,e^{[\delta\omega]}$；旋转向量把范数约束折进参数化（Diebel §7）。
- **网络输出：** 回归姿态用 6D（或 5D），推理时解码成 $R$，再用测地距离算损失；回归四元数时 $q$ 与 $-q$ 是同一旋转，损失需取 $\min(\|q-\hat q\|,\|q+\hat q\|)$（此条为工程惯例，Zhou 论文未给出）。
- **约定检查：** 欧拉角有 12 种序列（Diebel §8 目录），四元数 scalar 顺序因库而异，接口必须写明。

## 局限与风险

- 三参数表示必有奇异（欧拉角万向锁；指数坐标在 $\theta=\pi$ 与 $R=I$ 附近的对数公式数值敏感，*Modern Robotics* §3.2.3.3 / 附录 B.3），所以不要把某种 3 维表示当全局坐标。
- 6D 连续性换来的是冗余维度与正交化步骤；两列共线时解码不稳定（推测，Zhou 论文以「差集 $D$ 上无定义」表述）。
- **tan_norm** 不是文献通用名，参考轴固定，与 Zhou 的「任意前两列」不同，不要混用解码。
- 本页「推荐场景」栏是工程经验，非论文结论。

## 关联页面
- [旋转表示方法对比（SO(3)）](../comparisons/so3-rotation-representations.md) — 欧拉 / 矩阵 / 轴角 / 四元数 / so(3) / 6D / tan_norm 选型总表
- [单位四元数与 SO(3)](./unit-quaternion-so3.md) — 四元数专页（Diebel / Shoemake / MR Ch 3）
- [tan_norm 旋转观测表示](./tan-norm-rotation.md) — MimicKit / ProtoMotions 的 6D 观测编码与 decode 链
- [李群、李代数与刚体旋转](./lie-group-rigid-body-motions.md) — SO(3)/SE(3) 与 so(3)/se(3) 分工、四元数存储与 exp/log 优化链路
- [MimicKit](../entities/mimickit.md) — char_obs / tar_obs 中的 tan_norm 用法
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [SE(3) 切空间浮动基 TO](../entities/paper-se3-tangent-to.md) — TO 里浮动基「变量/差分/积分」三决策的对照实验
- [AHMP](../entities/paper-ahmp.md) — 同一切空间内层 + 接触发现
- [Visual Servoing](../methods/visual-servoing.md)
- [Action Tokenization](./vla-tokenization.md)
- [Modern Robotics 教材](../entities/modern-robotics-book.md) — Ch 3 用李群 / 螺旋理论系统建立 SO(3)/SE(3) 与 twist/wrench 的物理与数学语言

## 参考来源
- Lynch & Park, *Modern Robotics* (2017) §3.2–3.5、附录 B — [官方 PDF](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf)
- [Diebel 2006 姿态参数化统一参考](../../sources/papers/diebel_2006_representing_attitude_quaternions.md)
- [Shoemake 1985 四元数曲线与 SLERP](../../sources/papers/shoemake_1985_quaternion_curves_siggraph.md)
- [Zhou et al. CVPR 2019 连续旋转表示](../../sources/papers/zhou_2019_cvpr_continuity_rotation_representations.md)
- [MimicKit tan_norm 源码摘录](../../sources/repos/mimickit_tan_norm.md)
- Lynch, K. M., & Park, F. C. (2017). *Modern Robotics*. Ch 3 *Rigid-Body Motions* — SO(3)/SE(3) 的李群结构、指数映射、twist 表示。
- [sources/papers/perception.md](../../sources/papers/perception.md)
- [sources/papers/modern_robotics_textbook.md](../../sources/papers/modern_robotics_textbook.md)
- [深蓝具身智能：李群、李代数、四元数（微信公众号）](../../sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md) — 具身场景下四元数 / 李代数 / 6D 表示的分工直觉
- [SE(3) 切空间 TO 摘录（arXiv:2508.11520）](../../sources/papers/se3_tangent_to_arxiv_2508_11520.md) — 浮动基 TO 参数化对照

## 推荐继续阅读

- [Zhou et al., CVPR 2019 (arXiv:1812.07035)](https://arxiv.org/abs/1812.07035) — 连续旋转表示原文
- [Diebel 2006 attitude PDF](https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf) — 欧拉 / 四元数 / 旋转向量转换手册
- [Modern Robotics 官方 PDF](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf) — Ch 3 与附录 B
