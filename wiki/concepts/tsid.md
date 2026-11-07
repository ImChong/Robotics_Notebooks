---
type: concept
tags: [control, wbc, inverse-dynamics, optimization, humanoid]
status: complete
summary: "TSID 用任务空间逆动力学统一处理多任务和接触约束，是全身控制的重要方法页。"
updated: 2026-11-07
sources:
  - ../../sources/papers/delprete_tsid_arxiv_1410_3863.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
  - ../../sources/repos/tsid.md
  - ../../sources/papers/whole_body_control.md
  - ../../sources/papers/null_space_control.md
original:
  - ../../sources/papers/delprete_tsid_arxiv_1410_3863.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
  - ../../sources/repos/tsid.md
---

# TSID

**TSID（Task Space Inverse Dynamics，任务空间逆动力学）** 是一种典型的人形机器人全身控制方法，用来在满足动力学与接触约束的前提下，把任务空间目标转成可执行的关节加速度、力矩和接触力。

> **溯源提醒（2026-11-07 对照原文）**：Del Prete 等 2014 年提出 "Task Space Inverse Dynamics" 的原论文只处理**全驱动**机器人，用**伪逆 + 递归零空间投影**得到解析解，**没有 QP、没有不等式约束**，力矩可由 RNEA 以 $O(n)$ 算出。今天常说的"浮动基座 + 接触力 + 摩擦锥 + QP/HQP 的 TSID"来自后续工作：Saab 等 2013 的动态 HQP 与 `stack-of-tasks/tsid` 开源库。本页下文把两者分开写。

## 一句话定义

如果说 MPC / Centroidal Dynamics 在回答“机器人整体接下来该怎么动”，那 **TSID** 回答的就是：

> 这些目标具体怎么变成每个关节该出的力、该走的加速度，并且别把接触和平衡搞炸。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| HQP | Hierarchical Quadratic Programming | 按优先级分层堆叠任务与约束的二次规划 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| OCP | Optimal Control Problem | MPC 每步求解的有限时域最优控制问题 |

## 为什么重要

人形机器人控制最难的一步，不是写出一个漂亮的高层目标，而是把目标真正落到身体上。

比如你想同时做到：
- 躯干保持稳定
- 足端跟踪摆腿轨迹
- 手臂做动作
- 保持接触约束成立
- 关节力矩别超限

这些目标之间天然会冲突。

TSID 重要就在于：
- 它是 **WBC（Whole-Body Control）** 的代表性实现路线之一
- 它把任务空间目标、动力学约束、接触约束统一到一个优化框架里
- 它非常适合人形、双足、机械臂这类高自由度系统
- 它是“从规划参考到可执行全身控制”的关键落地点

一句话，**TSID 是把“我想让机器人这样动”变成“机器人真的能这样发力”的桥。**

## TSID 在解决什么问题

TSID 要解决的问题本质上是：

> 给定若干任务目标和物理约束，求一个满足动力学的一致控制解。

输出通常包括：
- 关节加速度 \( \ddot{q} \)
- 关节力矩 \( \tau \)
- 接触力 \( f \)

注意这和单纯 inverse kinematics 不一样。

- **IK** 只关心几何层怎么摆到目标位置
- **TSID** 关心动力学上能不能做到，以及力怎么出

## 核心原理

### 1. 在任务空间定义目标

常见任务包括：
- 躯干姿态跟踪
- 足端位置 / 姿态跟踪
- 手端轨迹跟踪
- CoM 跟踪
- 动量调节
- 关节姿态正则项

这些任务通常写成加速度层或力层目标，比如：

$$
\ddot{x}^* = \ddot{x}_{ref} + K_p (x_{ref} - x) + K_d (\dot{x}_{ref} - \dot{x})
$$

然后通过雅可比映射到广义坐标空间：

$$
\ddot{x} = J(q) \ddot{q} + \dot{J}(q, \dot{q}) \dot{q}
$$

也就是说，任务空间目标最后都会落到对 \( \ddot{q} \) 的约束或代价上。

### 2. 满足刚体动力学

机器人动力学方程通常写成：

$$
M(q) \ddot{q} + h(q, \dot{q}) = S^T \tau + J_c^T f
$$

其中：
- \( M(q) \)：质量矩阵
- \( h(q, \dot{q}) \)：重力、科里奥利、离心项
- \( \tau \)：关节力矩
- \( J_c \)：接触雅可比
- \( f \)：接触力
- \( S \)：驱动选择矩阵

TSID 的核心不是“猜一个动作”，而是要求这个动作必须符合真实动力学。

### 3. 满足接触约束

如果脚在地上，就不能乱滑、乱飞。

典型接触约束包括：
- 接触点加速度为零
- 法向力非负
- 摩擦锥约束
- 接触力矩限制

比如静接触约束：

$$
J_c \ddot{q} + \dot{J}_c \dot{q} = 0
$$

这保证接触点不会违反接触假设。

## 典型求解形式

同一个名字下有两代做法，务必分清（均已对照原文核实）：

### A. 原论文：全驱动机器人的解析解（Del Prete 2014）

从最高优先级任务 $N$ 到最低的姿态任务 0 依次求解"在不破坏更高优先级任务的前提下最小化本任务误差"，结果是（原文式 9–10，$W=I$ 时）：

$$
\tau^* = M\big(\ddot q_1 + N_{p(0)}\ddot q_p^*\big) + h,\qquad
\ddot q_i = \ddot q_{i+1} + (J_i N_{p(i)})^{\dagger}\big(\ddot x_i^* - \dot J_i\dot q - J_i \ddot q_{i+1}\big)
$$

其中 $\ddot q_p^*=K_p(q_p-q)-K_d\dot q$ 是姿态任务。要点：先在加速度层做带优先级的逆运动学，再一次逆动力学求力矩（运动学与动力学解耦），可用 RNEA 以 $O(n)$ 计算，无需显式质量矩阵；刚性接触约束作为最高优先级任务，力矩为 $\tau^*=M(\ldots)+h-J_c^Tf^*$。代价是**不含不等式约束**（关节限位、力矩界、摩擦锥都做不了）、只保证瞬时最优、且仅限全驱动。

### B. 浮动基座 + 接触 + 不等式：QP / HQP 形式（Saab 2013 与 TSID 库）

人形是欠驱动浮动基座，必须把接触力放进决策变量。Saab 等把决策变量取为 $u=(\tau,\ddot q,f)$，层级为"动力学方程 ≺ 接触不动 ≺ 法向力非负 ≺ 任务"；开源库 `tsid` 则取 $[\ddot q,\ f]$ 为变量，再由驱动行动力学事后解出 $\tau=M_a\ddot q+h_a-J_a^\top f$。

#### 单层加权 QP

把多个任务写成加权最小二乘：

$$
\min_{\ddot{q},\tau,f} \sum_i w_i \| A_i z - b_i \|^2
$$

同时满足：
- 动力学等式约束
- 接触约束
- 力矩 / 加速度 / 接触力边界

这里 $z$ 一般是联合决策变量，例如：

$$
z = [\ddot{q}, \tau, f]
$$

#### 分层 QP / Hierarchical QP

更经典的人形控制写法是优先级结构：

1. 最高优先级：动力学与接触约束
2. 次优先级：平衡 / 躯干 / 足端关键任务
3. 再次优先级：手臂动作、姿态优化、正则项

这样能处理"任务冲突"的问题。一句话就是：

> 先保命，再走路，再优雅。

**工程现实**：`stack-of-tasks/tsid` 库的 `addMotionTask(task, weight, priorityLevel)` 里，优先级 0 是硬约束层（基座动力学、接触、限位），其后各层是**带权重的最小二乘**；默认的 eiquadprog 求解器只接受**至多 2 层**（读源码 `solver-HQP-eiquadprog.cpp`：超过 2 层直接报错）。所以"多层严格 HQP"在该库里实际常被"硬约束层 + 一层加权任务"取代。严格多层 HQP 求解器见 [HQP](./hqp.md)。

## TSID 和 WBC 的关系

TSID 不是和 WBC 并列的东西，它基本可以看作 **WBC 的代表性实现方法之一**。

关系大概是：

- **WBC**：更大的范畴，指全身协调控制思想
- **TSID**：WBC 中非常经典的一条动力学一致优化路线

所以在人形控制里你常会看到：
- WBC 框架
- TSID 求解器
- HQP controller

它们很多时候是在一个技术栈上说不同层级的东西。

## TSID 和 Centroidal Dynamics 的关系

这俩很容易混，但其实分工不同。

### Centroidal Dynamics 更像上层
它关心：
- 质心怎么动
- 整体动量怎么演化
- 接触力大概怎么分布

### TSID 更像下层执行层
它关心：
- 具体关节怎么加速
- 每个驱动器怎么出力
- 接触约束如何被精确满足

很常见的分层结构是：

```text
Centroidal Planner / MPC
    ↓
给出 CoM / momentum / footstep / contact force reference
    ↓
TSID / WBC
    ↓
输出 joint acceleration / torque / contact force
```

这就是为什么你把 `Centroidal Dynamics` 写完后，下一页就该补 `TSID`，不然控制链条会断在中间。

## TSID 和 Inverse Kinematics / Inverse Dynamics 的区别

### 和 IK 的区别
- IK：只管几何位置是否能摆到
- TSID：管动力学、接触、力矩、任务冲突

### 和传统 Inverse Dynamics 的区别
- 传统 inverse dynamics 更像“给定加速度，求力矩”
- TSID 是“在多个任务和约束下，同时求一致的加速度、力矩、接触力”

所以 TSID 不是单条公式，而是一套任务化、约束化、优化化的 inverse dynamics 框架。

## 在机器人中的典型应用

### 1. 人形机器人步行控制
最常见：
- 躯干稳住
- 支撑脚保持接触
- 摆动脚跟踪轨迹
- CoM 跟踪参考
- 关节姿态做冗余优化

### 2. 双臂 / 全身操作
当机器人一边站着一边伸手操作时，TSID 很适合同时处理：
- 手端任务
- 身体姿态稳定
- 地面接触一致性

### 3. 受约束运动控制
凡是有接触、有闭链、有多任务优先级的问题，TSID 都很强。

> **证据边界（原文核实）**：Del Prete 2014 只在 23 自由度 CoMan 的**固定基座仿真**里验证（力控 + 双手 + 颈部任务，TSID 单周期 0.24 ms，对比 WBCF 0.64 ms）；Saab 2013 在 HRP-2 上做的是**离线**动态接触序列生成（"坐进扶手椅"、接触切换），轨迹再用位置控制回放，原文明确"尚未实时"。上面"步行控制""全身操作"是该方法族的常见用法，并非这两篇原文本身的实验结论（推测性概括）。

## 常见优点

- 动力学一致，不是只看几何
- 能自然处理多任务与优先级
- 能显式加入接触和摩擦约束
- 很适合人形 / 足式 / 高自由度系统
- 能和 MPC、centroidal planner 自然分层结合

## 常见局限

### 1. 依赖模型质量
动力学模型、接触模型、状态估计如果不准，TSID 效果会明显掉。

### 2. 实时求解压力
高频控制下解 QP/HQP 是有计算压力的，尤其任务多、约束多时。

### 3. 调参不轻松
任务权重、优先级、阻尼、正则项都需要经验，不是写完就飞。

### 4. 接触切换仍然麻烦
脚抬起、落地那一下最容易出幺蛾子，TSID 也得配合状态机、接触估计和过渡策略。Saab 2013 指出：若"法向力非负"这一层不可行（强接触不稳定），松弛变量非零、解在动力学上不一致，此时应改变接触模式；而这种情形往往出现在机器人已经在倒下的时候。

### 5. 原论文自述的三条局限（Del Prete 2014 §9.1）
- 不含不等式约束（关节限位、力矩界）；
- 只保证**瞬时（局部）最优**，不规划，任务可能把机器人带进更高优先级任务奇异的构型——作者建议把代价扩展到时域（MPC）；
- 仅针对全驱动机器人，浮动基座留给后续工作。

### 6. 严格层级 vs 工程实现
`stack-of-tasks/tsid` 默认求解器只支持 2 层（硬约束层 + 加权任务层），多层严格优先级需要专用 HQP 求解器（见 [HQP](./hqp.md)）。

## 工程实践

**用 `stack-of-tasks/tsid` 搭一个控制器的典型流程**（README 与源码核对，库版本 1.10.0，BSD-2-Clause，基于 Pinocchio + eiquadprog，另有 Python 绑定，`conda install tsid -c conda-forge` 即可安装）：

1. 用 Pinocchio 载入 URDF，建 `RobotWrapper`，建 `InverseDynamicsFormulationAccForce`；
2. `addRigidContact(contact, force_regularization_weight, ...)` 登记支撑脚（`Contact6d` 内部用线性化摩擦锥 + 法向力上下界）；
3. 用 `addMotionTask(task, weight, priorityLevel)` 加 CoM、躯干姿态、摆动脚、关节姿态（postural）任务；Del Prete 原文强调总要加一个**姿态任务**：它让解与权重矩阵无关，并稳定关节空间的零空间；
4. 每个控制周期 `computeProblemData(t, q, v)` 得到 QP，再用 `SolverHQuadProg`（可开 warm start、设最大迭代与最大求解时间）求解，最后读 `getActuatorForces / getAccelerations / getContactForces`；
5. 接触切换时用库提供的 `transition_duration` 参数在若干时间内平滑增删任务或接触。

**参数起点**（仅为论文仿真使用的值，不是通用推荐）：任务 PD 增益 $K_p=10\,s^{-2}$、$K_d=5\,s^{-1}$；阻尼伪逆系数 $\lambda=0.02$（最大增益 $(2\lambda)^{-1}=25$）、零空间奇异值阈值 $2.5\times10^{-8}$。

**验证建议**：先用库自带的 `exercises/`（机械臂 / 人形 / 四足 Python 例）和 `demo/demo_romeo.py` 跑通，再换自己的机器人；调参细节见 [WBC 调参指南](../queries/wbc-tuning-guide.md)，搭建流程见 [WBC 工程实现指南](../queries/wbc-implementation-guide.md)。

## 最小代码骨架

下面这段代码把 TSID 的最小味道抽出来：
- 先在任务空间定义一个期望加速度
- 再通过雅可比把任务落到广义加速度变量上
- 最后解一个最小二乘问题得到关节加速度命令

```python
import numpy as np

J = np.array([[1.0, 0.5], [0.0, 1.0]])
Jdot_qdot = np.array([0.0, 0.0])
acc_task_des = np.array([0.2, -0.1])

qdd_star, *_ = np.linalg.lstsq(J, acc_task_des - Jdot_qdot, rcond=None)
print("joint acceleration command:", qdd_star)
```

真实 TSID 会再加上：
- 动力学等式约束 `M(q) qdd + h = S^T tau + J_c^T f`
- 接触约束 `J_c qdd + Jdot_c qdot = 0`
- 力矩边界、摩擦锥、任务优先级

## 学这个方法时最应该盯住的三件事

1. **原理**：TSID 不是几何 IK，而是带动力学和接触约束的任务空间逆动力学
2. **最小代码**：你至少要能把 `xdd*`、`J qdd + Jdot qdot` 这层关系写成一个小例子
3. **局限性**：一旦模型、接触状态、任务权重不准，TSID 就会从“优雅协调”迅速退化成“数学上能解、机器人上难用”

## 参考来源

- [Del Prete et al. 2014 — TSID 原论文通读归档](../../sources/papers/delprete_tsid_arxiv_1410_3863.md) — *Prioritized motion-force control of constrained fully-actuated robots: "Task Space Inverse Dynamics"*（arXiv:1410.3863，RAS 2015）
- [Saab et al. 2013 — 刚性接触下动态全身运动生成通读归档](../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md) — 浮动基座 + 接触 + 不等式的 HQP 写法
- [TSID 仓库归档（README 与关键源码通读）](../../sources/repos/tsid.md) — 步骤 2.5：已开源（BSD-2-Clause）
- [TSID library](https://github.com/stack-of-tasks/tsid) — 开源实现
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md) — TSID/HQP 相关论文 ingest 摘要（二手整理）
- [零空间控制论文簇](../../sources/papers/null_space_control.md) — 任务优先级与零空间投影背景

## 关联页面

### 和 HQP 的关系
分层 QP（HQP）是处理多任务冲突的一类求解框架——严格层级保证高优先级任务精确满足。TSID 的原论文用零空间投影实现层级；`tsid` 库实践中多用"硬约束层 + 一层加权任务"（默认求解器至多 2 层），严格多层 HQP 需专用求解器（Escande 2014）。

见：[HQP（Hierarchical QP）](./hqp.md)

### 和 Whole-Body Control 的关系
TSID 是 WBC 最经典的实现路线之一，可以理解为”动力学一致的全身任务控制”。数学定义见 [Task Space Inverse Dynamics (TSID) 形式化](../formalizations/tsid-formulation.md)。

见：[Whole-Body Control](./whole-body-control.md)

### 和 Centroidal Dynamics 的关系
Centroidal Dynamics 常用来做上层平衡 / 动量 / 接触规划，TSID 负责把这些参考真正落成关节层控制。

见：[Centroidal Dynamics](./centroidal-dynamics.md)

### 和 Locomotion 的关系
在双足 locomotion 里，TSID 经常负责执行摆腿、稳躯干、保接触、跟踪 CoM 参考。

见：[Locomotion](../tasks/locomotion.md)

### 和 Optimal Control / MPC 的关系
MPC 负责规划未来参考，TSID 负责在每个控制周期把这些参考变成动力学可行的低层动作。

见：[Optimal Control (OCP)](./optimal-control.md)

见：[Model Predictive Control (MPC)](../methods/model-predictive-control.md)

## 继续深挖入口

如果你想沿着 TSID 继续往下挖，建议从这里进入：

### 运动学接口
- [逆运动学](../formalizations/inverse-kinematics.md) — 位置层 IK；TSID 是力/加速度升级
- [雅可比矩阵](../formalizations/robot-jacobian.md) — 任务约束投影到关节的 $J$
- [零空间控制](./null-space-control.md) — 单臂 7 轴显式投影；TSID 用 HQP 替代手写 $N$

### 论文入口
- [Whole-Body Control 论文导航](../../references/papers/whole-body-control.md)

### 工具 / Repo 入口
- [Utilities](../../references/repos/utilities.md)
- [Humanoid Projects](../../references/repos/humanoid-projects.md)

## 推荐继续阅读

- Del Prete et al., *Prioritized motion-force control of constrained fully-actuated robots: “Task Space Inverse Dynamics”*：<https://arxiv.org/abs/1410.3863>
- Saab et al., *Dynamic whole-body motion generation under rigid contacts and other unilateral constraints*：<https://hal-lirmm.ccsd.cnrs.fr/lirmm-00831097/document>
- TSID library: <https://github.com/stack-of-tasks/tsid>
- Andrea Del Prete 的 TSID 教学幻灯片与视频：<https://andreadelprete.github.io/#teaching>（README 推荐；本次未逐讲核对）

## 一句话记忆

> TSID 是把任务空间目标、动力学约束、接触约束和力矩分配统一起来的全身控制方法，它是人形机器人控制从“想这么动”到“真能这么动”的关键落地点。
