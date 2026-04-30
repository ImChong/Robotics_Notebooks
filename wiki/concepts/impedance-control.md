---
type: concept
tags: [control, manipulation, impedance-control, force-control, contact-rich, whole-body-control]
status: complete
updated: 2026-04-20
summary: "Impedance Control 通过把末端行为写成质量-弹簧-阻尼关系，让机器人在接触任务中既能跟踪目标又能保持柔顺。"
sources:
  - ../../sources/papers/contact_dynamics.md
  - ../../sources/papers/contact_planning.md
related:
  - ./contact-rich-manipulation.md
  - ./whole-body-control.md
  - ./tsid.md
  - ../methods/tactile-impedance-control.md
  - ../tasks/manipulation.md
  - ../queries/contact-rich-manipulation-guide.md
---

# Impedance Control（阻抗控制）

**阻抗控制**：不直接要求机器人“精确走到某个位姿”，而是规定当机器人与环境之间出现位置误差或接触力时，系统应该表现出怎样的 **质量-弹簧-阻尼（Mass-Spring-Damper）** 响应。

## 一句话定义

阻抗控制的核心不是把误差压到零，而是预先设计好“**误差会产生什么力、以多快速度衰减**”。

## 为什么重要

在机器人与物理环境发生交互（如插拔、装配、抛光）时，纯位置控制极其危险：即使是 1mm 的示教误差，在刚性环境下也会产生巨大的接触力，导致硬件损坏。

阻抗控制通过将机器人末端建模为一个虚拟的物理系统，赋予了机器人 **主动柔顺性（Active Compliance）**。这使得机器人在碰到障碍物时不是“硬顶”，而是像弹簧一样退让，并在撤去外力后恢复目标位置。它是当前具身智能（VLA/IL）输出动作指令后，底层执行器最稳健的接口。

## 核心数学模型

在任务空间（Task Space）中，期望的阻抗行为通常描述为：
$$ M_d (\ddot{x} - \ddot{x}_d) + B_d (\dot{x} - \dot{x}_d) + K_d (x - x_d) = f_{ext} $$
其中：
- $x, x_d$：当前位姿与期望位姿。
- $M_d, B_d, K_d$：**期望的质量、阻尼、刚度矩阵**。这是控制器的核心参数。
- $f_{ext}$：机器人末端受到的外部环境力。

### 简化形式（准静态）
在低速操作中，通常忽略质量项，阻抗控制简化为力映射：
$$ f = K_d (x_d - x) + B_d (\dot{x}_d - \dot{x}) $$
然后通过雅可比转置映射到关节力矩：$\tau = J^T f$。

## 阻抗控制 vs. 导纳控制（Admittance Control）

两者都是为了实现阻抗行为，但实现路径相反：

- **阻抗控制（Impedance Control）**：
  - **输入**：位移误差（Position/Velocity）。
  - **输出**：力/力矩（Force/Torque）。
  - **实现**：通常需要机器人支持力矩控制（Torque-controlled），直接调节电机输出。
  - **优点**：响应极快，可以处理高频碰撞。

- **导纳控制（Admittance Control）**：
  - **输入**：外部力（Force/Torque）。
  - **输出**：修正后的位移目标（Modified Position）。
  - **实现**：在标准位置控制机器人上通过外接力传感器实现。
  - **优点**：适合大负载、刚性强的工业机器人，实现简单。

## 在人形机器人中的应用

1. **足端阻抗**：在步行过程中，足端阻抗控制可以吸收地形不平带来的冲击力，起到虚拟减震器的作用，提高平衡鲁棒性。
2. **双臂协同操作**：当两只手共同搬运一个刚性物体时，两臂的阻抗控制可以吸收相互之间的挤压力，防止因几何闭链约束过紧而导致的电机过热或保护停机。
3. **安全人机交互**：当人类推搡机器人时，阻抗控制允许机器人顺着外力方向移动，确保不会对人造成二次伤害。

## 调参建议（Tuning Tips）

- **刚度 $K_d$**：决定了机器人的“硬度”。
  - 任务需要高精度（如搬运）时调高。
  - 任务涉及未知接触（如摸索孔位）时调低。
- **阻尼 $B_d$**：决定了运动的平稳性。
  - 通常设置为 **临界阻尼** 状态（$B_d \approx 2\sqrt{M_d K_d}$），以防止系统在受到扰动后产生长久的往复振荡。
- **奇异点处理**：在接近机构奇异点时，雅可比转置映射可能失效，需要增加正则项或限制最大输出力。

## 常见误区

- **误区 1：刚度越大越稳。** 事实上，环境越硬，机器人刚度必须越低，否则系统闭环特征值会移向不稳定区。
- **误区 2：阻抗控制需要精确环境模型。** 阻抗控制的初衷就是为了应对环境的不确定性，它不需要知道环境的具体几何信息。

## 参考来源

- Hogan, N. (1985). *Impedance Control: An Approach to Manipulation*. Journal of Dynamic Systems, Measurement, and Control.
- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — 接触力与柔顺执行基础
- [sources/papers/contact_planning.md](../../sources/papers/contact_planning.md) — 接触任务中的执行层组织

## 关联页面

- [Contact-Rich Manipulation](./contact-rich-manipulation.md)
- [Whole-Body Control](./whole-body-control.md)
- [Force Control Basics (力控制基础)](./force-control-basics.md) — 阻抗控制的理论背景
- [Tactile Impedance Control](../methods/tactile-impedance-control.md) — 由触觉信号在线驱动 $K_d, B_d$ 的变参数推广
- [TSID](./tsid.md)
- [Manipulation](../tasks/manipulation.md)
- [Query：接触丰富操作实践指南](../queries/contact-rich-manipulation-guide.md)
