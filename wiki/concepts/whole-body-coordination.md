---
type: concept
tags: [whole-body, coordination, humanoid, locomotion, manipulation, loco-manipulation, control]
status: stub
summary: "全身协调控制（Whole-Body Coordination）关注多肢体、多链接系统在运动和操作中的时空同步问题，是 WBC 的上层概念，在 loco-manipulation 中尤为关键。"
sources:
  - ../../sources/papers/whole_body_control.md
related:
  - ./whole-body-control.md
  - ../tasks/loco-manipulation.md
  - ../tasks/bimanual-manipulation.md
  - ./centroidal-dynamics.md
  - ./contact-dynamics.md
---

# Whole-Body Coordination（全身协调控制）

**全身协调控制（Whole-Body Coordination）**：研究高自由度机器人系统（尤其是人形机器人）如何将全身多个肢体、链接的运动在时间和空间上进行统一协调，使不同子系统的运动相互配合，共同实现全身层面的任务目标。

## 一句话定义

全身协调控制关注的是"**让整个身体作为一个整体工作**"——不仅仅是让每个关节独立运动，而是确保腿部、躯干、手臂的运动在动力学上相互协调、在任务层面相互支持、在时序上精准配合。

## 为什么重要

### 从生物学视角

人类的全身协调能力来自数百万年的进化：行走时手臂自然摆动（平衡角动量），投掷时全身旋转（传递动量），搬重物时躯干前倾（调整质心）。这些协调模式深植于神经系统，不需要有意识控制。

机器人缺乏这种内置协调机制——每一种协调行为都需要被明确建模和控制。

### 从工程视角

人形机器人的协调问题在以下方面特别突出：

1. **动力学耦合**：手臂运动改变全身质心位置和角动量，直接影响下肢平衡。若不考虑这种耦合，单独优化手臂和腿的运动会导致整体不稳定。

2. **冗余度利用**：30+ 自由度的人形机器人有大量冗余。全身协调需要有目的地利用冗余（如用躯干运动辅助手臂到达更远目标）而非随意化解。

3. **多目标冲突的调解**：行走任务（保持平衡、跟踪速度）和操作任务（追踪目标、精确抓取）可能对质心轨迹和接触力提出相互冲突的需求，协调控制需要系统地解决这些冲突。

4. **接触拓扑变化**：行走时接触点不断切换（双支撑→单支撑→飞行），每次切换都改变了系统的约束结构，全身协调必须适应这种动态变化。

## 核心概念

### 1. 运动层级（Motion Hierarchy）

全身协调通常通过任务优先级框架实现：

```
层级 1（最高优先级）：重心/质心控制（保持平衡）
层级 2：接触力/步态控制（维持稳定接触）
层级 3：任务空间运动（末端执行器轨迹）
层级 4：关节姿态保持（避免奇异配置）
```

高优先级任务的零空间（null space）被低优先级任务用来完成次要目标，互不干扰。

### 2. 角动量管理（Angular Momentum Regulation）

手臂运动会产生角动量，干扰躯干姿态和行走稳定性。全身协调的关键之一是**主动管理全身角动量**：

$$\dot{H}_{CoM} = \sum_i (r_i - r_{CoM}) \times f_i^{contact}$$

机器人可以通过手臂运动产生"反向"角动量来补偿腿部步态引起的扰动（如人类行走时手臂自然前后摆动的原理）。

### 3. 冗余度解析（Redundancy Resolution）

当任务约束不足以完全确定全身运动时，冗余度提供了额外自由度。全身协调通过以下方式利用冗余：

- **次要任务（Secondary Tasks）**：在主任务零空间内最小化关节运动、避免奇异构型、优化灵活性
- **能量优化**：选择最小化关节力矩的全身配置
- **预整姿（Pre-shaping）**：提前调整躯干和肢体配置，为下一步操作做准备

### 4. 全身质心协调

全身协调的最基础约束来自质心动力学：

$$m\ddot{r}_{CoM} = \sum_i f_i^{contact} + mg$$

$$\dot{H}_{CoM} = \sum_i \tau_i^{contact}$$

无论执行什么上层任务，全身运动必须满足这个质心动力学方程，这是全身协调的物理底线。

## 全身协调 vs Whole-Body Control (WBC)

**WBC 和全身协调是什么关系？**

| 维度 | WBC | 全身协调 |
|------|-----|---------|
| 层级 | 控制实现层 | 概念/目标层 |
| 关注点 | 如何求解关节力矩（优化/QP） | 多肢体运动的时空配合原则 |
| 输出 | 具体的关节命令 | 设计原则和架构指导 |
| 典型工具 | TSID、HQP、Crocoddyl | 概念框架，无固定实现 |

WBC 是实现全身协调的**具体技术手段**；全身协调是 WBC 需要达到的**目标语义**。一个 WBC 控制器可能存在技术上可行但"协调感"很差的解（如不自然的手臂摆动），全身协调研究关注的正是这类"自然性"和"效率性"。

## 在 Loco-Manipulation 中的作用

Loco-Manipulation（边走边操作）是全身协调最复杂的场景：

### 动力学耦合的具体体现

```
[操作任务] → 手臂需要施力 → 手臂运动/接触力 
         → 改变全身角动量 → 影响步态稳定性
         → 需要腿部补偿   → 影响行走速度和轨迹
         → 反馈到操作精度（行走抖动影响手臂末端）
```

这个耦合循环说明，在 loco-manipulation 中，操作精度和行走稳定性是**相互制约**的，不能独立优化。

### 协调策略

**策略1：独立控制 + 前馈补偿**
- 行走控制器 + 操作控制器独立运行
- 操作控制器通过前馈将手臂运动引起的质心扰动补偿给行走控制器
- 简单但补偿精度有限

**策略2：统一优化（Unified WBC）**
- 将行走任务和操作任务纳入同一 QP 优化
- 质心约束自动调解两者的冲突
- 理论最优但计算复杂

**策略3：分层协调**
- 质心层：维持全身平衡（高优先级）
- 步态层：执行行走规划（中优先级）
- 操作层：执行末端轨迹（低优先级，在零空间内）

## 学习方法中的全身协调

传统方法通过显式建模实现协调，学习方法则试图从数据中隐式学习：

### 模仿学习路线

- **全身动作捕获**：记录人类在操作时的完整全身运动（含行走和双臂动作）
- **运动重定向（Motion Retargeting）**：将人类运动映射到机器人关节空间，保留协调结构
- **策略学习**：从重定向后的演示数据中学习全身协调策略

挑战：人机形态差异（比例、关节范围、质量分布）使运动重定向精度有限，协调结构可能在重定向中扭曲。

### 强化学习路线

- 设计奖励函数同时激励行走质量和操作精度
- 自动发现协调策略，但训练困难、协调行为可能不自然
- 可以通过参考运动（reference motion）引导协调姿态

## 常见误区

1. **协调 ≠ 同步**：全身协调不是让所有关节同时运动，而是让不同关节在**各自适当的时机**运动，并相互配合。

2. **高自由度不等于高协调**：自由度多提供了更多可能性，但不自动产生协调——协调需要显式设计或学习。

3. **局部优化的陷阱**：独立优化手臂轨迹和腿部步态，再把结果叠加，几乎总是次优的。真正的全身协调需要全局考量。

## 参考来源

- Sentis & Khatib, *Synthesis of Whole-Body Behaviors Through Hierarchical Control of Behavioral Primitives* — 全身协调层级控制奠基工作
- Cheng et al., *Expressive Whole-Body Control for Humanoid Robots* (2024) — 学习方法实现全身协调
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md) — WBC / TSID / HQP 技术实现背景

## 关联页面

- [Whole-Body Control (WBC)](./whole-body-control.md) — 全身协调的核心实现工具
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 全身协调最复杂的应用场景
- [Bimanual Manipulation](../tasks/bimanual-manipulation.md) — 双臂协调是全身协调的子问题
- [Centroidal Dynamics](./centroidal-dynamics.md) — 全身协调的物理基础
- [Contact Dynamics](./contact-dynamics.md) — 接触力管理是全身协调的关键约束

## 推荐继续阅读

- Cheng et al., [*Expressive Whole-Body Control for Humanoid Robots*](https://arxiv.org/abs/2402.16796)
- [TSID](./tsid.md) — 全身协调控制的核心实现框架
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 全身协调的终极挑战

## 一句话记忆

> 全身协调控制不是让每个关节独立跑，而是让整个身体像一个有机体一样运作——腿走路时手臂自然摆动，手臂操作时躯干自动补偿，所有运动在动力学层面互相支持而非互相干扰。
