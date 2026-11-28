# Coordination and Decentralized Cooperation of Multiple Mobile Manipulators（Khatib et al., JRS 1996）

> 来源归档（paper）

- **标题：** Coordination and Decentralized Cooperation of Multiple Mobile Manipulators
- **作者：** Oussama Khatib、Kazuhiro Yokoi、Kyong-Sok Chang、Diego Ruspini、Robert Holmberg、Arancha Casal、Andreas Baader
- **机构：** Stanford University（Robotics Laboratory, Department of Computer Science）；Kazuhiro Yokoi 访问自 MITI / MEL（日本筑波）
- **期刊：** *Journal of Robotic Systems*, Vol. 13, No. 11, pp. 755–764 (1996)
- **PDF 原文：** <https://cs.stanford.edu/groups/manips/publications/pdfs/Khatib_1996_JRS.pdf>
- **入库日期：** 2026-11-29
- **核对范围：** 已获取 Stanford CS 原文 PDF 并通读全文 §1–§5（Introduction、Vehicle/Arm Coordination、Augmented Object Model、Virtual Linkage、Decentralized Cooperation、Stanford Romeo/Jules 双全向移动操作臂物理实验，共 772 行文本）。
- **通读：** 已完成
- **开源状态：** **不适用 / 历史期刊论文（无官方开源仓库）**

## 核心原文摘录

### 1) 全身操作空间动力学（Vehicle/Arm Coordination）

- 移动操作臂由高惯量、低带宽移动底座与高带宽、低惯量机械臂串联，呈现显著**动力学非对称（dynamic asymmetry）**与冗余自由度。
- 将整机在任务空间构建为统一动态模型 $\Lambda(q)\ddot{x} + \mu(q,\dot{q}) + p(q) = F$，并通过动力学一致零空间（dynamically consistent null-space $N(q)^T \Gamma_0$）协调移动底座定位与操作臂精细交互：当操作臂端处于高频力/运动控制时，由广义零空间扭矩引导移动底座进入高可操作度（manipulability）位形。

### 2) 增广物体模型（Augmented Object Model）

- 对多台机器人共同刚性夹持同一负载 $m_o$ 的系统，无需重新推导整体多臂联合大矩阵，而是将**各机器人末端操作空间动力学**叠加到**负载本体刚体动力学**上：
  $$\Lambda_{\oplus}(q)\ddot{x} + \mu_{\oplus}(q,\dot{q}) + p_{\oplus}(q) = F_{\oplus}$$
  其中 $\Lambda_{\oplus}(q) = \Lambda_o + \sum_{i=1}^N \Lambda_i(q)$，总操作力为各机施加到物体坐标系的总和 $F_{\oplus} = \sum_{i=1}^N F_i$，实现模块化的运动学/惯量叠加（additive operational space inertial properties）。

### 3) 虚拟连杆模型（Virtual Linkage）与去中心化协作（Decentralized Cooperation）

- 多机紧耦合闭链搬运的**内力（internal forces）**与**物体净运动（external wrench）**通过 $N$ 节点刚体连接的 **Virtual Linkage** 显式正交解耦：
  - **集中式 Virtual Linkage** 需要全局测量相对位姿并统一求解 $W$（grasp/internal force matrix）分配；
  - **Decentralized Cooperation（§4）**：以某一台机器人（或单机自身端坐标系）为主导输出物体期望加速度与轨迹，其余从机通过**力传感/导纳或零运动刚度约束（zero-motion / contact wrench sensing）**维持固定抓持几何并对物体动力学施加**负载分担比例 $\alpha_i$**（使从机自主承担自身局部增广惯量与指定比例的物体惯量），从而在不传递高带宽实时关节状态广播的条件下完成协同搬运，并有效抑制刚性闭链挤压内力。

### 4) 物理实验验证（§5）

- 在 Stanford **Romeo & Jules** 两台全向移动底座（Holonomic-Mobile Platform）配 PUMA 560 机械臂平台上验证：实现双机无通信协同搬运、动态擦拭黑板（需控制法向接触力与切向轨迹）、以及人机牵引共搬等全身/多机任务。

## 对 wiki 的映射

- [多机器人协作（Multi-Robot Collaboration）](../../wiki/concepts/multi-robot-collaboration.md)
- [协同物体搬运（Collaborative Object Transport）](../../wiki/concepts/collaborative-object-transport.md)
- [多机器人协同架构对比](../../wiki/comparisons/multi-robot-coordination-architectures.md)
