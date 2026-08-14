# 重力补偿算法一手论文簇

> 来源归档（ingest）

- **标题：** Gravity compensation for robot manipulators
- **类型：** paper（经典簇：模型基 $g(q)$、PD+重力、迭代学习、自适应重力）
- **来源：** ASME JDSMC / IJACSP / IEEE T-RA / IJRR
- **入库日期：** 2026-08-13
- **最后更新：** 2026-08-13
- **一句话说明：** 把「抵消重力广义力 $g(q)$」从 RNEA 计算、PD 调节、未知负载学习到惯量辨识收成一条控制用法，而不是再写一遍逆动力学教材。
- **沉淀到 wiki：** 是 → [`wiki/concepts/gravity-compensation.md`](../../wiki/concepts/gravity-compensation.md)

## 开源状态（步骤 2.5）

本簇经典论文**无官方控制器仓**。可运行实现以动力学库与厂商教程为准：

| 资料 | 代码 | 结论 |
|------|------|------|
| Takegaki & Arimoto 1981 | 公式被广泛复现，无原作者仓 | **确认未开源** |
| De Luca & Panzieri 1993 | 实验室柔性臂实验，无官方仓 | **确认未开源**（开放 PDF） |
| Tomei 1991 | 3-DoF 仿真，无官方仓 | **确认未开源**（教学 PDF） |
| Luh / Walker / Paul 1980；Featherstone RNEA | 算法进教材与库 | 计算入口见 [Pinocchio](../repos/pinocchio.md) / [Dynibo](../repos/dynibo.md) |
| PAL TIAGo 教程 | [gravity_compensation_controller_tutorial](https://github.com/pal-robotics/gravity_compensation_controller_tutorial) | **部分开源**：教程仓可运行；`package.xml` 许可证为 TODO、GitHub `license=null`；生产控制器 `pal_controllers/GravityCompensationController` 在 PAL OS **未开源** |
| Pinocchio / Dynibo | `computeGeneralizedGravity` / `gravity()` | **已开源**（已有仓库归档，不重复造实体） |

## 核心论文摘录（MVP）

### 1) A New Feedback Method for Dynamic Control of Manipulators（Takegaki & Arimoto, 1981）

- **链接：** <https://doi.org/10.1115/1.3139651>
- **核心贡献：** 从力学/能量观点证明：关节坐标的 **线性 PD 反馈** 对大范围运动有效；在重力场下，调节问题的标准做法是 **PD + 重力补偿**。这是后续「PD+$g(q)$ / PD+$g(q_d)$」全局稳定分析的源头，而不是一套新的动力学递推。
- **对 wiki 的映射：**
  - [重力补偿](../../wiki/concepts/gravity-compensation.md)
  - [PID 控制](../../wiki/methods/pid-control.md)

### 2) On-Line Computational Scheme for Mechanical Manipulators（Luh, Walker, Paul, 1980）

- **链接：** <https://doi.org/10.1115/1.3149599>
- **核心贡献：** 把牛顿–欧拉递推写成可在线计算的逆动力学。静止特例

$$
g(q)=\mathrm{RNEA}(q,0,0)
$$

是所有模型基重力补偿的计算核；现代实现是 Pinocchio / RBDL / Dynibo，而不是手写 $n$ 个连杆的拉格朗日展开。
- **对 wiki 的映射：**
  - [重力补偿](../../wiki/concepts/gravity-compensation.md)
  - [ABA / RNEA](../../wiki/formalizations/articulated-body-algorithms.md)

### 3) Learning gravity compensation in robots: Rigid arms, elastic joints, flexible links（De Luca & Panzieri, 1993）

- **链接：** DOI <https://doi.org/10.1002/acs.4480070510>；开放 PDF <https://www.diag.uniroma1.it/~labrob/pub/papers/IJACSP93.pdf>
- **核心贡献：** 未知负载或 $g(q)$ 估不准时，**不要**指望一次模型补偿或整段 PID。用关节 PD + **离散更新的常值前馈**：每轮到达稳态后把当前控制量收成下一轮的 $\hat g(q_d)$。刚臂、弹性关节、柔性杆在刚度压过重力的条件下全局收敛。三连杆刚臂仿真 + 带柔性前臂的两连杆真机（倾斜平面引入重力）。
- **对 wiki 的映射：**
  - [迭代学习重力补偿（论文实体）](../../wiki/entities/paper-learning-gravity-compensation.md)
  - [重力补偿](../../wiki/concepts/gravity-compensation.md)

### 4) Adaptive PD controller for robot manipulators（Tomei, 1991）

- **链接：** <https://doi.org/10.1109/70.86088>；教学 PDF <http://www.diag.uniroma1.it/~deluca/rob2_en/AdaptivePDgravity_Tomei.pdf>
- **核心贡献：** 重力参数（含负载）未知时，**连续自适应**估计线性回归形式的重力项，叠在 PD 上；点到点全局收敛，并给出跟踪扩展。与 De Luca 的差别：Tomei 是在线参数自适应，De Luca 是设定点之间的迭代前馈，二者都不需要完整 $M(q)$。
- **对 wiki 的映射：**
  - [重力补偿](../../wiki/concepts/gravity-compensation.md)
  - [System Identification](../../wiki/concepts/system-identification.md)

### 5) Estimation of inertial parameters of manipulator loads and links（Atkeson, An, Hollerbach, 1986）

- **链接：** <https://doi.org/10.1177/027836498600500306>
- **核心贡献：** 用关节力矩/力测量辨识连杆与负载的惯性参数，从而得到可用的 $g(q)$。模型基重力补偿的精度上限由这类 SysID 决定，而不是由 RNEA 公式决定。
- **对 wiki 的映射：**
  - [重力补偿](../../wiki/concepts/gravity-compensation.md)
  - [System Identification](../../wiki/concepts/system-identification.md)
  - [连杆与转子惯量](../../wiki/concepts/robot-link-and-rotor-inertia.md)
