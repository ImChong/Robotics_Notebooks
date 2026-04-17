---
type: concept
tags: [locomotion, control, dynamics, bipedal]
status: complete
summary: "LIP/ZMP 用简化倒立摆和零力矩点刻画双足稳定性，是经典 humanoid 行走控制的理论基础。"
---

# LIP / ZMP

**LIP（Linear Inverted Pendulum, 线性倒立摆）** 和 **ZMP（Zero Moment Point, 零力矩点）** 是双足机器人行走控制里最经典的一对基础模型与稳定性概念。

## 一句话定义

- **LIP**：用“质心高度近似不变”的简化模型描述双足机器人平衡与步行。
- **ZMP**：地面对机器人合力产生的等效力矩为零时，对应落在支撑面内的那个点。

一句话说白了：

> LIP 给你一个够简单、还能算得动的行走动力学模型，ZMP 给你一个判断“会不会倒”的稳定性指标。

## 为什么重要

在人形机器人控制里，完整动力学太复杂，直接上全模型做实时规划代价很高。

LIP / ZMP 之所以经典，不是因为它们最精确，而是因为它们在“够简单”和“够有用”之间取得了非常好的平衡：

- 能把双足行走问题先降维到质心运动
- 能把稳定性问题转成可计算约束
- 是很多步态规划、MPC、预览控制的理论起点
- 即使今天有 centroidal dynamics、NMPC、RL，它们仍然是理解 locomotion 的入门主干

## LIP 的核心假设

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

## 在机器人里的作用

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
日本 Honda / HRP 系路线里非常经典。

思路是：
- 提前看未来几步的参考 ZMP
- 通过预览控制生成更平滑、更稳定的 CoM 轨迹

### 2. 3D-LIPM
从单平面扩展到二维水平运动，是最常见版本。

### 3. LIPM + Footstep Planning
把落脚点也加入优化，就能做更实用的步态规划。

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

## 常见局限

### 1. 质心高度固定是假设，不是真实世界
真实人形机器人走路时，质心高度会变化。

### 2. 忽略角动量
手臂摆动、躯干调整等对平衡其实很重要，LIP 往往没法完整表达。

### 3. 不适合高动态动作
跑、跳、快速扰动恢复时，LIP 的近似通常不够。

### 4. ZMP 不是唯一稳定性指标
对于更动态的动作，常常会用：
- Capture Point
- Divergent Component of Motion (DCM)
- Viability / Reachability
- Full-body contact feasibility

所以别把 ZMP 神化。它很经典，但不是终点。

## 参考来源

- Kajita et al., *Introduction to Humanoid Robotics* — LIP/ZMP 理论基础经典教材
- Kajita et al., *Biped walking pattern generation by using preview control of zero-moment point* — preview control 经典论文
- Wieber, *Trajectory Free Linear Model Predictive Control for Stable Walking in the Presence of Strong Perturbations* — LIP 与 MPC 的理论连接
- **ingest 档案：** [sources/papers/footstep_and_balance.md](../../sources/papers/footstep_and_balance.md) — ZMP / CP / DCM / Herdt 在线步位规划

## 和已有页面的关系

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

## 推荐继续阅读

- Kajita et al., *Introduction to Humanoid Robotics*
- Kajita et al., *Biped walking pattern generation by using preview control of zero-moment point*
- Wieber, *Trajectory Free Linear Model Predictive Control for Stable Walking in the Presence of Strong Perturbations*

## 一句话记忆

> LIP 是“简化的人形行走动力学”，ZMP 是“经典的步行稳定性指标”，两者一起构成了双足机器人行走控制的入门主干。
