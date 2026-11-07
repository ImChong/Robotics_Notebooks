---
type: concept
tags: [dynamics, locomotion, control, mpc, wbc]
status: complete
related:
  - ./floating-base-dynamics.md
  - ./whole-body-control.md
  - ./lip-zmp.md
  - ../methods/model-predictive-control.md
  - ../entities/paper-adp.md
  - ../entities/paper-adaptive-whole-body-mpc-payload.md
original:
  - ../../sources/papers/herzog_2016_momentum_control_hierarchical_inverse_dynamics_arxiv_1410_7284.md
sources:
  - ../../sources/papers/herzog_2016_momentum_control_hierarchical_inverse_dynamics_arxiv_1410_7284.md
  - ../../sources/papers/mpc.md
  - ../../sources/papers/whole_body_control.md
  - ../../sources/papers/adp_arxiv_2607_03454.md
  - ../../sources/papers/adaptive_whole_body_mpc_2610_12972.md
summary: "Centroidal Dynamics 以质心动量为核心描述全身运动，是 MPC、WBC 和多接触规划的重要中间层。"
updated: 2026-11-07
---

# Centroidal Dynamics

**Centroidal Dynamics（质心动力学）**：用机器人整体质心的线动量和角动量来描述全身动力学的一种中层建模方式。

## 一句话定义

如果说全关节动力学太重、LIP 又太简，那么 **Centroidal Dynamics** 就是中间那层最常用的折中：

> 不追着每个关节细节打，而是先抓住“整个机器人质心怎么动、整体角动量怎么变”。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| LIP | Linear Inverted Pendulum | 线性倒立摆，质心动力学的常用简化模型 |
| ZMP | Zero Moment Point | 足式平衡判据，地面反力合力矩为零的点 |
| ANYmal | ANYbotics Quadruped | ANYbotics 的四足机器人研究平台 |
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| OCP | Optimal Control Problem | MPC 每步求解的有限时域最优控制问题 |

## 为什么重要

人形机器人控制里最烦的地方在于：
- 自由度高
- 接触切换频繁
- 平衡约束强
- 实时优化计算压力大

这时候你通常不想一上来就对全关节刚体动力学做重型在线优化，但又不能像 LIP 那样简化得太狠。

Centroidal Dynamics 重要就在这：

- 比 LIP 更真实，能表达角动量和接触力作用
- 比全关节动力学更轻，适合做实时规划和 MPC
- 是 locomotion、balance、contact planning、WBC 的核心中间层
- 很多人形/四足控制器其实都在用这一层做上层规划

一句话，**它是“机器人整体怎么站住、怎么发力、怎么挪质心”的主干模型。**

## 核心原理：质心动量与动力学方程

### 核心对象：质心动量

Centroidal Dynamics 关注的不是每个关节变量本身，而是机器人整体的：

- **线动量** \( \mathbf{l} = m \dot{\mathbf{c}} \)
- **角动量** \( \mathbf{k} \)

其中：
- \( m \)：总质量
- \( \mathbf{c} \)：质心位置
- \( \dot{\mathbf{c}} \)：质心速度

把线动量和角动量合起来，可以写成 centroidal momentum：

$$
\mathbf{h} =
\begin{bmatrix}
\mathbf{k} \\
\mathbf{l}
\end{bmatrix}
$$

（排序因文献而异：Herzog et al. 2016 写 $h=[h_{lin};h_{ang}]$，线动量在前，均取在质心。）

这玩意儿本质上是在说：

> 别先问每个关节怎么动，先问整台机器人整体的动量怎么演化。

### 核心动力学方程

在有多个接触点时，质心动力学通常可写成：

### 线动量变化

$$
\dot{\mathbf{l}} = \sum_i \mathbf{f}_i + m \mathbf{g}
$$

意思很直白：
- 所有接触力加起来
- 再加上重力
- 决定整体线动量怎么变

### 角动量变化

$$
\dot{\mathbf{k}} = \sum_i (\mathbf{p}_i - \mathbf{c}) \times \mathbf{f}_i + \sum_i \boldsymbol{\tau}_i
$$

这里：
- \( \mathbf{p}_i \)：接触点位置
- \( \mathbf{f}_i \)：接触力
- \( \boldsymbol{\tau}_i \)：接触力矩
- \( \mathbf{c} \)：质心位置

这个式子说明：
- 接触力不只是推着机器人平移
- 还会通过力臂影响整体角动量

这正是它比 LIP 强很多的地方。

**这两个式子是怎么来的（Herzog et al. 2016 的推导）：** 浮动基机器人的运动方程 $M\ddot q+N=S^\top\tau+J_c^\top\lambda$ 中，对应 6 个未驱动基座自由度的最后 6 行不含关节力矩，就是整机牛顿–欧拉方程：质心动量变化率 = 外力（接触力 + 重力）。写成矩阵形式即
$$
\dot h=\begin{bmatrix}I&0&\cdots\\ [x_i-x_{cog}]_\times&I&\cdots\end{bmatrix}\lambda+\begin{bmatrix}mg\\0\end{bmatrix},\qquad \dot x_{cog}=\frac{1}{m}h_{lin}
$$
其中 $x_i$ 为第 $i$ 个接触点、$\lambda$ 为广义接触力（力 + 力矩）。所以在已知接触力时，**这个层次本身是精确的**，不是近似；近似发生在把惯量 / 构型耦合 $h=H_G\dot q$ 用简化模型代替的时候（推测：如单刚体模型）。详见 [Floating Base Dynamics](./floating-base-dynamics.md)。

## 和 LIP / ZMP 的关系

LIP / ZMP 可以看成是 Centroidal Dynamics 的强简化版本。

### LIP 做了哪些简化
- 质心高度近似恒定
- 角动量近似忽略
- 接触模式很简单
- 动力学线性化得很狠

所以：

- **LIP / ZMP** 更适合入门理解和线性步态规划
- **Centroidal Dynamics** 更适合真实的人形/四足平衡和接触规划

关系可以理解成：

```text
全关节刚体动力学
    ↓（抽掉很多细节）
Centroidal Dynamics
    ↓（再进一步简化）
LIP / ZMP
```

## 为什么它适合机器人控制

### 1. 抓住平衡的主干
对 locomotion 来说，最关键的其实不是每个关节微小细节，而是：
- 质心往哪走
- 接触力怎么分配
- 整体角动量会不会炸

Centroidal Dynamics 正好抓住这几个核心量。

### 2. 适合做接触力优化
你可以直接把接触力当决策变量，做：
- 平衡维持
- 步态规划
- 扰动恢复
- 多接触动作规划

### 3. 适合 MPC / Convex MPC
很多足式机器人控制器就是：
- 上层用 centroidal model 做 MPC
- 下层用 WBC / inverse dynamics 跟踪

这是工程里非常常见、也非常靠谱的分层方案。

## 工程实践：典型用法

### 1. 足式机器人平衡与行走
最经典用途：
- 规划未来几步质心轨迹
- 优化地面反作用力
- 保证摩擦锥、支撑约束、力界限满足

常见于：
- 双足行走
- 四足 trot / pace / bound
- 扰动恢复控制

### 2. Convex MPC
很多 MIT Cheetah / ANYmal / 人形 locomotion 控制器都会用 centroidal dynamics 的离散形式来做凸优化（此条为通行说法，本批未逐篇核对这些系统）。

因为它比全刚体动力学更适合实时求解。

### 3. Whole-Body Control 的上层参考
WBC 不一定自己负责高层规划。

很常见的架构是：

```text
Centroidal Planner / MPC
    ↓
生成 CoM / momentum / contact force 参考
    ↓
Whole-Body Controller
    ↓
关节加速度 / 力矩 / 接触力分配
```

### 4. 扰动恢复
机器人被推一下后，是否能稳住，本质上常常取决于：
- 质心速度
- 角动量变化
- 接触力还能不能救回来

这正是 centroidal level 最擅长分析的东西。

## Centroidal Momentum Matrix（CMM）

这是一个很重要但也最容易把人绕晕的概念。

机器人整体动量可以写成：

$$
\mathbf{h} = \mathbf{A}_g(\mathbf{q}) \dot{\mathbf{q}},\qquad \dot{\mathbf h}=\mathbf A_g\ddot{\mathbf q}+\dot{\mathbf A}_g\dot{\mathbf q}
$$

其中：
- \( \mathbf{A}_g(\mathbf{q}) \) 是 **Centroidal Momentum Matrix**
- \( \mathbf{q} \) 是广义坐标
- \( \dot{\mathbf{q}} \) 是广义速度

它的意义是：

> 给定当前姿态和关节速度，CMM 告诉你整台机器人的整体动量是多少。

Herzog et al. 称它为 centroidal momentum matrix $H_G$，引自 Orin & Goswami (IROS 2008)（本批未取得该原文）。动量变化率因此既可当**运动学任务**（$\ddot q$ 的函数，需要 $\dot H_G$，常由数值微分得到而放大噪声），也可当**力任务**（$\lambda$ 的函数）；作者主张用力形式设计动量控制律，并对它线性化后做 LQR，利用线 / 角动量耦合，避免在各接触构型下手调 PD 增益。

这在：
- momentum control
- inverse dynamics
- task formulation
- WBC
里都非常常见。

## 常见约束

做 centroidal planning 时，常见约束包括：

- **接触力约束**：法向力非负
- **摩擦锥约束**：防止打滑
- **支撑区域约束**：接触点必须合理
- **力矩/动作边界**：防止不现实的接触解
- **接触时序约束**：哪只脚何时可用

这些约束比 LIP / ZMP 框架更自然。

## 常见局限

### 0. 来自原文的实测限制
Herzog et al. 在 Sarcos 液压人形上用该层 + 层级逆动力学做平衡 / 跟踪实验，作者在结论中指出：模型不准与计算复杂度仍是现实约束；他们的方法需要接更高层规划器才能行走（未来工作）。控制器要求每步物理一致（运动方程 + 力矩饱和）作为最高优先级，其余任务才在其零空间里满足。

### 1. 还是忽略了很多关节级细节
Centroidal Dynamics 比 LIP 强，但它依然不是全刚体动力学。

它不直接保证：
- 每个关节都可达
- 姿态细节都合理
- 自碰撞一定避免

所以它常常只是上层，不是全部。

### 2. 需要下层控制器兜底
上层规划出一个很漂亮的 centroidal 轨迹，不代表机器人关节层一定能实现。

所以通常需要：
- WBC
- inverse dynamics
- joint-level tracking

去把它落地。

### 3. 接触建模仍然麻烦
接触什么时候建立、什么时候断开、冲击怎么处理，这些仍然很难。

Centroidal 模型没有魔法，只是让问题更可控，不是让问题消失。

## 最小代码骨架

这段代码只保留 centroidal level 最核心的量：
- 质心线动量变化由接触力和重力决定
- 不去管每个关节怎么动
- 先建立“整体怎么被地面推着走”的直觉

```python
import numpy as np

m = 40.0
g = np.array([0.0, 0.0, -9.81])
forces = [
    np.array([20.0, 0.0, 220.0]),
    np.array([-15.0, 0.0, 210.0]),
]

l_dot = sum(forces) + m * g
com_acc = l_dot / m
print("centroidal linear momentum rate:", l_dot)
print("com acceleration:", com_acc)
```

如果你把这里的 `forces` 当作优化变量，再加上摩擦锥、支撑约束和预测时域，基本就走到了 centroidal MPC 的入口。

## 学这个方法时最应该盯住的三件事

1. **原理**：它研究的是整体线动量 / 角动量，不是每个关节的细节动力学
2. **最小代码**：你至少要能自己写出“接触力 → 线动量变化 → 质心加速度”的最小推演
3. **局限性**：它是中层模型，不会自动保证关节可达、姿态自然、接触切换平滑，所以下一层通常必须接 WBC / TSID

## 参考来源

- [Herzog et al. 2016，arXiv:1410.7284 原文归档](../../sources/papers/herzog_2016_momentum_control_hierarchical_inverse_dynamics_arxiv_1410_7284.md) — 已通读：质心动量矩阵、动量变化率的力 / 运动学两种写法、LQR 动量控制
- Orin, Goswami, Lee, *Centroidal dynamics of a humanoid robot* — centroidal dynamics 系统性阐述（本批未取得原文，未核对）
- Herzog et al., *Momentum Control with Hierarchical Inverse Dynamics on a Torque-Controlled Humanoid* — WBC 与 centroidal 结合代表
- Winkler et al., *Gait and Trajectory Optimization for Legged Systems through Phase-based End-Effector Parameterization* — 接触规划代表（未核对）

## 关联页面（和已有页面的关系）

### 和 LIP / ZMP 的关系
LIP / ZMP 是更简化的双足平衡模型，Centroidal Dynamics 则是更接近真实机器人的中层动力学。

见：[LIP / ZMP](./lip-zmp.md)

### 和 MPC 的关系
Centroidal Dynamics 是很多足式 / 人形 MPC 控制器最常见的预测模型。

见：[Model Predictive Control (MPC)](../methods/model-predictive-control.md)

[未知载荷下的自适应全身 MPC](../entities/paper-adaptive-whole-body-mpc-payload.md)（arXiv:2610.12972）以 **质心 MPC** 生成动力学可行参考，再由自适应模糊平衡控制器补偿未知载荷引入的模型失配——说明质心模型参数（质量/惯量）偏差需要下游补偿层兜底。

[MPC-RL](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) 将 **CD-MPC** 用于 **训练期** landmark guidance reward（非部署期在线 MPC），是 centroidal 模型进入大规模 RL 的一条 2026 路线。

### 和 Whole-Body Control 的关系
很多 WBC 架构里，上层用 centroidal planner 给参考，下层再做全身优化执行。

见：[Whole-Body Control](./whole-body-control.md)

### 和 Locomotion 的关系
只要是研究平衡、步态、接触力分配、扰动恢复，Centroidal Dynamics 基本绕不过去。

见：[Locomotion](../tasks/locomotion.md)

[ADP](../entities/paper-adp.md) 用 **SRBD 轨迹优化** 生成质心/接触力参考，再以对抗奖励约束策略动力学时间窗——是质心层模型进入 **AMP 式先验** 的一条 2026 路线。

### 和 Optimal Control 的关系
把质心轨迹、动量变化、接触力和约束一起写进优化问题，本质上就是一个 centroidal-level 的最优控制问题。

见：[Optimal Control (OCP)](./optimal-control.md)

## 继续深挖入口

如果你想沿着 centroidal dynamics 继续往下挖，建议从这里进入：

### 论文入口
- [Whole-Body Control 论文导航](../../references/papers/whole-body-control.md)
- [Survey Papers](../../references/papers/survey-papers.md)

### 开源项目 / 工具入口
- [Humanoid Projects](../../references/repos/humanoid-projects.md)
- [Utilities](../../references/repos/utilities.md)

## 推荐继续阅读

- [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) — 本页所述物理/仿真要素在保真度链路（建模 ① → 数值 ② → 接触 ③ → 随机化 ④）中的定位
- Orin, Goswami, Lee, *Centroidal dynamics of a humanoid robot*
- [Herzog et al., *Momentum Control with Hierarchical Inverse Dynamics on a Torque-Controlled Humanoid* (arXiv:1410.7284)](https://arxiv.org/abs/1410.7284)
- [接触力旋量锥](../formalizations/contact-wrench-cone.md) — 把摩擦锥推广到 6 维 wrench，作为质心层的接触约束
- Winkler et al., *Gait and Trajectory Optimization for Legged Systems through Phase-based End-Effector Parameterization*

## 一句话记忆

> Centroidal Dynamics 关心的是“整个机器人质心和整体动量怎么演化”，它是连接 LIP / ZMP 与全身控制、MPC、真实足式机器人规划之间的关键中层模型。
