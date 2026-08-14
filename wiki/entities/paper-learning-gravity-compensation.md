---
type: entity
tags: [paper, control, gravity, adaptive-control, sapienza]
status: complete
updated: 2026-08-13
venue: "IJACSP 1993"
related:
  - ../concepts/gravity-compensation.md
  - ../methods/pid-control.md
  - ../concepts/system-identification.md
  - ../concepts/friction-compensation.md
sources:
  - ../../sources/papers/de_luca_learning_gravity_compensation_1993.md
  - ../../sources/papers/gravity_compensation.md
summary: "De Luca / Panzieri IJACSP 1993：未知重力时用关节 PD + 离散更新的常值前馈学习设定点补偿；刚臂、弹性关节、柔性杆均可全局收敛。"
---

# 迭代学习重力补偿（De Luca & Panzieri, 1993）

**De Luca, Panzieri** 的 *Learning gravity compensation in robots: Rigid arms, elastic joints, flexible links*（[IJACSP 1993](https://doi.org/10.1002/acs.4480070510)，[开放 PDF](https://www.diag.uniroma1.it/~labrob/pub/papers/IJACSP93.pdf)）把「不知道 $g(q)$ 或负载变了」从 PID 调参问题改写成一个**有证明的迭代前馈**。

## 一句话定义

**先用关节 PD 走到一个会停住但有静差的点，把稳态力矩收成下一轮的常值重力前馈，直到设定点误差收缩到零。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PD | Proportional–Derivative | 每轮迭代的反馈核，不加积分 |
| FF | Feedforward | 离散更新的常值 $\hat u_i$，目标是 $g(q_d)$ |
| GAS | Global Asymptotic Stability | 文中在增益条件下对任意初值成立 |
| SEA | Series Elastic Actuator | 弹性关节是本文覆盖的第二类对象 |
| DoF | Degrees of Freedom | 仿真 3 连杆；真机 2 连杆柔性前臂 |

## 为什么重要

[重力补偿](../concepts/gravity-compensation.md) 的默认实现是模型基 $g(q)$。抓未知工件、URDF 质量不准、或柔性引起的额外下垂时，那一项是错的。PID 能消静差但只有刚臂的**局部**证明，大行程还 windup。本文给出第三条路：**不要模型因式分解、不要高增益、不要连续积分**，只在设定点之间更新一个常值。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 罗马第一大学（Sapienza）DIAG LabRob |
| **平台** | 三连杆刚臂仿真；两连杆柔性前臂真机（倾斜平面引入重力） |
| **开源** | **确认未开源**（无官方仓） |
| **对照实现** | 模型基路线用 [Pinocchio](../entities/pinocchio.md) `computeGeneralizedGravity`；真机示教用 [PAL 教程](../../sources/repos/gravity-compensation-controller-tutorial.md)（许可未声明） |

## 核心原理 / 方法栈

已知重力时的 PD+：

$$
u = K_P(q_d-q)-K_D\dot q + g(q_d)
$$

$K_P$ 需压过重力梯度界 $\alpha$（$\|\partial g/\partial q\|\le\alpha$）。柔性时还要求结构刚度压过重力，保证平衡唯一。

未知重力时第 $i$ 轮：

$$
u = \frac{K_P}{\beta}(q_d-q)-K_D\dot q + \hat u_{i-1}
$$

$\hat u_0=0$（或用现有 $\hat g(q_d)$ 热启动）。到达稳态后，未知的 $g(q_i)$ 就等于当前控制量；下一轮前馈取

$$
\hat u_i = K_P(q_d-q_i)+\hat u_{i-1}
$$

刚臂上收缩条件可收成 $\lambda_{\min}(K_P/\beta)>2\alpha$。柔性杆还要刚度条件。**没有显式积分状态**：更新只在「已经停住」的离散时刻发生，因此避开 PID windup。

```mermaid
flowchart LR
  PD["关节 PD"]
  FF["常值 û"]
  ARM["臂到达稳态 qi"]
  UPD["û ← û + Kp (qd − qi)"]
  PD --> ARM
  FF --> ARM
  ARM --> UPD --> FF
```

## 实验与评测

- **仿真：** 三连杆刚臂竖直面设定点；迭代后关节误差收到零，前馈收敛到该点重力。
- **真机：** LabRob 两连杆轻型臂、柔性前臂；平面倾斜以引入重力。高增益 PD 消不掉的柔性下垂，由迭代前馈补上。
- **不包含：** 轨迹跟踪（那是 Tomei 1991 的扩展方向）；没有开源复现包。

## 结论

**一句话总判：未知负载的设定点调节，优先用「PD + 离散更新前馈」，而不是加大 $K_i$ 或假装 URDF 质量是对的。**

1. **先确认是设定点问题** — 跟踪仍应走模型基 $g(q)$ 或 Tomei 自适应，本文不替代 CTC。
2. **$K_P$ 必须压过重力梯度** — 否则每轮平衡不唯一，迭代会漂。
3. **等停住再更新 $\hat u$** — 在运动中当积分用会破坏证明，也容易振荡。
4. **柔性臂多一截下垂** — 关节 PD 看得到电机侧误差，看不到杆挠度；本文把这两层一起补。
5. **有哪怕很差的 $\hat g(q_d)$ 也可以当 $\hat u_0$** — 少迭代几轮，不是必须从 0 开始。
6. **要复现计算核走开源库** — 本篇无代码；RNEA 用 Pinocchio/Dynibo，示教用 PAL 教程并自行确认许可。

## 源码运行时序图

**不适用**（1993 年实验室实验，无官方训练/推理/控制器仓）。模型基 $g(q)$ 的运行时序见 [重力补偿](../concepts/gravity-compensation.md) 中 PAL/Pinocchio 一节。

## 工程实践

| 检查项 | 建议 |
|--------|------|
| 适用 | 抓取未知工件后的点到点放置；柔性/弹性关节的稳态下垂 |
| 不要用 | 高速轨迹、需要连续补偿的步行（用 RNEA + WBC） |
| 停止判据 | $\|q_d-q\|$ 与 $\|\dot q\|$ 同时低于门限再更新 $\hat u$ |
| 与摩擦 | 静摩擦会偏置「稳态力矩=重力」；先补偿摩擦或只在低摩擦关节上用 |

## 局限与风险

- 每换一个设定点都要重新迭代；不是把整个 $g(q)$ 学成函数。
- 证明依赖重力 Lipschitz 界与刚度条件；接近奇异或负载剧烈晃动时界会破。
- 真机只有柔性前臂 2 连杆，不能外推到 7 轴摩擦主导的协作臂而不做摩擦处理。

## 与其他工作对比

| 工作 | 关系 |
|------|------|
| Takegaki & Arimoto 1981 | PD+重力的能量证明；本文在**未知** $g$ 时接棒 |
| Tomei 1991 | 连续自适应重力参数；本文是离散前馈，不要回归矩阵 |
| PID | 刚臂局部稳定、柔性无全局证明；本文明确对标并避开 I 项 |
| Atkeson 1986 SysID | 先辨识惯性再算 $g(q)$；负载频繁更换时迭代学习更便宜 |
| PAL / Pinocchio | 模型基工程实现；负载已知时不必上本文 |

## 关联页面

- [重力补偿](../concepts/gravity-compensation.md)
- [PID Control](../methods/pid-control.md)
- [System Identification](../concepts/system-identification.md)
- [Friction Compensation](../concepts/friction-compensation.md)
- [Pinocchio](../entities/pinocchio.md)

## 参考来源

- [De Luca & Panzieri 1993 归档](../../sources/papers/de_luca_learning_gravity_compensation_1993.md)
- [重力补偿论文簇](../../sources/papers/gravity_compensation.md)

## 推荐继续阅读

- 开放 PDF：<https://www.diag.uniroma1.it/~labrob/pub/papers/IJACSP93.pdf>
- Tomei, *Adaptive PD controller for robot manipulators*, IEEE T-RA 1991：<http://www.diag.uniroma1.it/~deluca/rob2_en/AdaptivePDgravity_Tomei.pdf>
