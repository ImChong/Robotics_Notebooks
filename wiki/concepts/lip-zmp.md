---
type: concept
tags: [locomotion, control, dynamics, bipedal]
status: complete
summary: "LIP/ZMP 用简化倒立摆和零力矩点刻画双足稳定性，是经典 humanoid 行走控制的理论基础。"
updated: 2026-11-07
original:
  - ../../sources/papers/lip_zmp_dcm_walking_reading_primary_papers.md
sources:
  - ../../sources/papers/lip_zmp_dcm_walking_reading_primary_papers.md
  - ../../sources/papers/footstep_and_balance.md
  - ../../sources/papers/humoslope_arxiv_2607_07830.md
related:
  - ../tasks/locomotion.md
  - ../methods/model-predictive-control.md
  - ./whole-body-control.md
  - ./optimal-control.md
  - ../entities/paper-humoslope-physics-guided-slope-locomotion.md
---

# LIP / ZMP

**LIP（Linear Inverted Pendulum, 线性倒立摆）** 和 **ZMP（Zero Moment Point, 零力矩点）** 是双足机器人行走控制里最经典的一对基础模型与稳定性概念。

## 一句话定义

- **LIP**：用“质心高度近似不变”的简化模型描述双足机器人平衡与步行。
- **ZMP**：地面对机器人合力产生的等效力矩为零时，对应落在支撑面内的那个点。

一句话说白了：

> LIP 给你一个够简单、还能算得动的行走动力学模型，ZMP 给你一个判断“会不会倒”的稳定性指标。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LIP | Linear Inverted Pendulum | 线性倒立摆，质心动力学的常用简化模型 |
| ZMP | Zero Moment Point | 足式平衡判据，地面反力合力矩为零的点 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| LQR | Linear Quadratic Regulator | 线性系统二次型代价下的最优反馈控制器 |
| DCM | Divergent Component of Motion | 质心发散分量，用于落脚点与平衡调节 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| OCP | Optimal Control Problem | MPC 每步求解的有限时域最优控制问题 |

## 为什么重要

在人形机器人控制里，完整动力学太复杂，直接上全模型做实时规划代价很高。

LIP / ZMP 之所以经典，不是因为它们最精确，而是因为它们在“够简单”和“够有用”之间取得了非常好的平衡：

- 能把双足行走问题先降维到质心运动
- 能把稳定性问题转成可计算约束
- 是很多步态规划、MPC、预览控制的理论起点
- 即使今天有 centroidal dynamics、NMPC、RL，它们仍然是理解 locomotion 的入门主干

## 核心原理：LIP 的核心假设

LIP 模型的典型假设：

1. 机器人质量集中在质心（CoM）
2. 质心高度近似恒定
3. 支撑脚与地面理想接触
4. 忽略角动量变化或把其影响简化掉

这样，复杂的人形机器人被近似成一个“放在支点上的线性倒立摆”。

### 经典二维形式

在质心高度固定为 \( z_c \) 时，水平面动力学可以写成：

$$
\ddot{x} = \omega^2 (x - x_{zmp}), \quad \omega = \sqrt{g / z_c}
$$

同理在 \( y \) 方向：

$$
\ddot{y} = \omega^2 (y - y_{zmp})
$$

这里：
- \( x, y \)：质心水平位置
- \( x_{zmp}, y_{zmp} \)：ZMP 位置
- \( g \)：重力加速度
- \( z_c \)：质心高度

这个式子很关键，因为它把“质心运动”和“支撑稳定性”联系起来了。

**核对原文后的补充（Herdt 等 2010；Romualdi 等 2018）：**

- 反过来，质心运动决定 ZMP：$x_{zmp} \approx x - \frac{z_c}{g}\ddot{x}$（Herdt 式 (5) 把它写作 CoP 的近似，忽略各肢体转动惯性效应）。ZMP 因此是“质心加速度的线性函数”，这是 LIP 能写成线性状态空间、进而做 LQR / 线性 MPC 的原因。
- 状态用质心位置与速度时，系统有一个正实特征根 $+\omega$，$x_{zmp}$ 固定就会指数发散；用 DCM $\xi = x + \dot{x}/\omega$ 换元后可拆成 $\dot{x} = \omega(\xi - x)$（稳定）与 $\dot{\xi} = \omega(\xi - x_{zmp})$（不稳定）两支，详见 [Capture Point / DCM](./capture-point-dcm.md)。
- Herdt 等的线性 MPC 预测模型是把质心位置 / 速度 / 加速度作状态、**jerk 分段常值**作输入的三阶积分器：$\hat{x}_{k+1}=A\hat{x}_k+B\dddot{x}_k$，ZMP 作输出 $z_k=[1\;0\;-h/g]\hat{x}_k$（采样 $T=0.1\,\mathrm{s}$，$N=16$ 步，预测 1.6 s，约两个步长）。

## ZMP 是什么

ZMP 的定义是：

> 地面对机器人合接触力产生的水平力矩在该点处为零的支撑面点。

更直觉地说：
- 如果 ZMP 落在支撑多边形内，机器人通常有机会保持动态平衡
- 如果 ZMP 跑出支撑区，系统就很容易翻倒或者必须快速跨步补偿

### 支撑多边形

对于双足机器人：
- 单脚支撑时，支撑多边形就是那只脚的接触区域
- 双脚支撑时，支撑多边形是两脚接触区域的凸包

经典稳定条件可以粗暴记成：

$$
ZMP \in \text{support polygon}
$$

这不是所有情况下都严格充分必要，但在经典步行控制里非常有用。

## LIP 和 ZMP 的关系

LIP 不是稳定性指标，ZMP 也不是动力学模型。

它们的关系是：

- **LIP** 提供简化动力学
- **ZMP** 提供稳定性约束
- 把两者合起来，就能把双足步行规划写成一个可优化、可控制的问题

典型链路：

```text
期望步态 / 足步规划
    ↓
规划 ZMP 轨迹
    ↓
根据 LIP 动力学求 CoM 轨迹
    ↓
下层控制器执行（MPC / WBC / inverse dynamics）
```

## 工程实践：在机器人里怎么用

### 1. 双足步行规划入门模型
很多早期人形机器人行走控制，核心就是：
- 先设计 ZMP 轨迹
- 再反推 CoM 轨迹
- 最后由关节控制器跟踪

### 2. MPC / Preview Control 的基础
LIP + ZMP 是经典步态 MPC 和预览控制的起点。

因为它足够线性，能写成状态空间形式，便于：
- LQR
- Preview Control
- Linear MPC

### 3. 帮你建立 locomotion 的直觉
就算以后转向 centroidal dynamics、NMPC、RL，人还是得先搞懂：
- 为什么前倾会导致必须迈步
- 为什么支撑面决定稳定余地
- 为什么质心轨迹不能乱来

这些底层直觉，LIP / ZMP 是最好的入门桥。

## 经典扩展

### 1. Preview Control
AIST / HRP 系路线里非常经典（Kajita 等 2003）。据 Herdt 等的转述，它是一个 LQR 方案：基于 LIP 的线性近似，在有限预测时域内最小化质心 jerk，同时让 ZMP 跟踪放在支撑区中间的参考，从而能在线生成行走并持续吸收当前状态；但它忽略对接触力的精确约束，面对速度指令快速变化或强扰动时能力有限。（Kajita 2003 原文本次未能取得，此处为二手转述。）

思路是：
- 提前看未来几步的参考 ZMP
- 通过预览控制生成更平滑、更稳定的 CoM 轨迹

### 1b. 线性 MPC 与自动落脚（Herdt 等 2010）
Wieber 2006 把 ZMP 边界作为显式约束，把 LQR 变成线性 MPC；Herdt 等进一步把**落脚点**也作为 QP 决策变量，只给参考速度，由 QP 自动决定步位：HRP-2 仿真中每 0.8 s 一步，完整 QP 平均求解时间 < 1 ms（QL / qpOASES），CoP 离支撑边界保留 3 cm 余量；被推或速度指令不可实现时“安全优先”——宁可少跟踪速度也保持 CoP 可行。已知代价：不允许交叉腿时横向速度每两步只有一步能被调准，平均横向速度只有指令的约 2/3。

### 2. 3D-LIPM
从单平面扩展到二维水平运动，是最常见版本（Kajita 2001，原文本次未取得；本页水平面 $x$、$y$ 解耦方程见上文）。

### 3. LIPM + Footstep Planning
把落脚点也加入优化，就能做更实用的步态规划。Khadiv 等 2016 进一步把**步时**也变成变量：用 $\tau = e^{\omega T}$ 换元使方程线性，仅看下一步就得到小规模 QP；60 kg、质心 0.8 m 的 LIPM 在 325 N、0.1 s 的侧推下，固定步时不能恢复，联合调整步位 + 步时可以恢复。点足 / 被动踝机器人无法靠 CoP 调节 DCM，这时迈步是更有效的稳定手段。

### 4. 从 LIP 走向 Centroidal Dynamics
LIP 太简化，忽略了：
- 角动量变化
- 质心高度变化
- 更复杂接触模式

所以更进一步通常会转向：
- Centroidal Dynamics
- Convex MPC
- NMPC
- Whole-Body Control

## 局限与常见风险：常见局限

### 1. 质心高度固定是假设，不是真实世界
真实人形机器人走路时，质心高度会变化。

### 2. 忽略角动量
手臂摆动、躯干调整等对平衡其实很重要，LIP 往往没法完整表达。

### 3. 不适合高动态动作
跑、跳、快速扰动恢复时，LIP 的近似通常不够。

### 3b. ZMP 约束只管接触可行性
Romualdi 等 2018 的 iCub 对比显示：即便采用带 ZMP 多边形约束的 DCM 预测控制，最高直行速度（0.19–0.22 m/s）也不如无约束的瞬时 DCM 控制（0.41 m/s）；作者的解释是预测控制产生的期望 ZMP 变化快，对 ZMP 跟踪误差更不鲁棒。可见“满足 ZMP 约束的最优解”并不自动等于“实机更稳”，执行层（力矩 / 位置控制、传感噪声）同样关键。

### 4. ZMP 不是唯一稳定性指标
对于更动态的动作，常常会用：
- Capture Point
- Divergent Component of Motion (DCM)
- Viability / Reachability
- Full-body contact feasibility

所以别把 ZMP 神化。它很经典，但不是终点。

### 5. 陡坡上别默认世界水平面参考

多数 RL/规划里的 ZMP 偏差在 **世界水平面** 上评估；坡度变大时，支撑平面与重力方向错位，水平面偏差会系统性偏离真实接触几何。[HumoSlope](../entities/paper-humoslope-physics-guided-slope-locomotion.md) 把正则改到**局部倾斜支撑平面**，并用接触力加权锚点做稠密奖励——这是把经典 ZMP 嵌进 model-free RL 时更物理一致的做法。

## 最小代码骨架

这段代码展示了 LIP / ZMP 的最小闭环：
- 先给一个目标 ZMP
- 再用简化动力学前向积分 CoM 状态
- 观察 ZMP 固定时 CoM 如何偏离（这是一个不稳定系统）

```python
import numpy as np

z_c = 0.8
g = 9.81
omega = np.sqrt(g / z_c)
dt = 0.01
x, dx = 0.05, 0.0   # CoM 位置与速度
x_zmp = 0.0         # 目标 ZMP 设在脚掌中心

traj = []
for _ in range(200):
    ddx = omega**2 * (x - x_zmp)
    dx += ddx * dt
    x += dx * dt
    traj.append((x, dx, x_zmp))

print("final com state:", traj[-1])
```

注意：该开环仿真里 `x_zmp` 固定为 0 而 CoM 初始偏离 5 cm，$\ddot{x}=\omega^2 x$ 会让 CoM **指数发散**（时间常数 $1/\omega\approx 0.29\,\mathrm{s}$），它演示的是 LIP 的不稳定性本身，而不是“被拉回”。真正做步态规划时，要把 `x_zmp` 设计成随时间变化的参考轨迹（或由反馈律给出），再反推出一条有界的 CoM 轨迹。

## 学这个方法时最应该盯住的三件事

1. **原理**：LIP 是简化动力学，ZMP 是稳定性约束，不要把两者混成一个东西
2. **最小代码**：你至少要能自己写出 `ddot{x} = ω²(x - x_zmp)` 的仿真 loop
3. **局限性**：一旦涉及高动态动作、显著质心起伏、复杂接触，就该往 centroidal dynamics / MPC / WBC 升级

## 参考来源

- **原文通读归档：** [LIP / ZMP / DCM 四篇一手论文通读](../../sources/papers/lip_zmp_dcm_walking_reading_primary_papers.md) — Herdt 等 2010、Romualdi 等 2018、Shafiee 等 2019、Khadiv 等 2016，全文已读
- Kajita et al., *Introduction to Humanoid Robotics* — LIP/ZMP 理论基础经典教材
- Kajita et al., *Biped walking pattern generation by using preview control of zero-moment point* — preview control 经典论文
- Wieber, *Trajectory Free Linear Model Predictive Control for Stable Walking in the Presence of Strong Perturbations* — LIP 与 MPC 的理论连接
- **ingest 档案：** [sources/papers/footstep_and_balance.md](../../sources/papers/footstep_and_balance.md) — ZMP / CP / DCM / Herdt 在线步位规划
- [HumoSlope（arXiv:2607.07830）](../../sources/papers/humoslope_arxiv_2607_07830.md) — 局部支撑平面 slope-adaptive ZMP 正则 + BSGA；G1 盲坡面 locomotion

## 关联页面

### 和 Locomotion 的关系
LIP / ZMP 是理解双足 locomotion 最基础的一层。

见：[Locomotion](../tasks/locomotion.md)

### 和 MPC 的关系
很多经典双足行走控制器，本质上就是基于 LIP 的线性 MPC 或 preview control。

见：[Model Predictive Control (MPC)](../methods/model-predictive-control.md)

### 和 Whole-Body Control 的关系
LIP / ZMP 常用于上层步态规划，WBC 负责下层把这些目标变成全身关节力矩和接触力分配。

见：[Whole-Body Control](./whole-body-control.md)

### 和 Optimal Control 的关系
当你把 ZMP 约束、CoM 轨迹、落脚点规划写进代价函数和约束里，本质上就是一个简化最优控制问题。

见：[Optimal Control (OCP)](./optimal-control.md)

### 和 HumoSlope 的关系
HumoSlope 把 ZMP 从水平面参考改写为**局部支撑平面偏差奖励**，再接生物力学门控以抑制蹲姿退化——是本页理论在陡坡 RL 中的直接延伸。

见：[HumoSlope 极端坡面物理引导步态](../entities/paper-humoslope-physics-guided-slope-locomotion.md)

## 推荐继续阅读

- [Capture Point / DCM](./capture-point-dcm.md) — 由 LIP 发散分量导出的下一步落脚判据
- [ZMP + LIP 形式化](../formalizations/zmp-lip.md) — 同一模型的形式化推导页
- [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) — 本页所述物理/仿真要素在保真度链路（建模 ① → 数值 ② → 接触 ③ → 随机化 ④）中的定位
- Kajita et al., *Introduction to Humanoid Robotics*
- Kajita et al., *Biped walking pattern generation by using preview control of zero-moment point*
- Wieber, *Trajectory Free Linear Model Predictive Control for Stable Walking in the Presence of Strong Perturbations*
- Herdt et al., *Online Walking Motion Generation with Automatic Foot Step Placement* (Advanced Robotics, 2010) — 开放版 <https://inria.hal.science/inria-00391408/document>，LIP 线性 MPC 的完整推导与 HRP-2 仿真

## 一句话记忆

> LIP 是“简化的人形行走动力学”，ZMP 是“经典的步行稳定性指标”，两者一起构成了双足机器人行走控制的入门主干。
