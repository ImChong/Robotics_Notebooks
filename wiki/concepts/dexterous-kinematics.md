---
type: concept
tags: [dexterity, kinematics, robot-hand, math, manipulation]
status: complete
updated: 2026-05-12
related:
  - ./humanoid-parallel-joint-kinematics.md
  - ./tactile-sensing.md
  - ../entities/allegro-hand.md
  - ../methods/in-hand-reorientation.md
  - ../formalizations/friction-cone.md
sources:
  - ../../sources/papers/contact_dynamics.md
summary: "灵巧手运动学（Dexterous Kinematics）研究多指末端执行器与被操作物体之间的运动映射关系，其核心在于处理多点接触形成的闭链约束。"
---

# Dexterous Kinematics (灵巧手运动学)

**灵巧手运动学 (Dexterous Kinematics)** 是机器人学中研究多指协同操作的理论基础。与传统的单臂串联运动学不同，灵巧手在抓取物体时，多个手指通过接触点与物体共同构成了一个**闭链运动学系统**。

## 核心数学模型

在灵巧手操作中，我们需要处理三个坐标系之间的变换：基座坐标系 $\{B\}$、手指末端坐标系 $\{F_i\}$ 和物体坐标系 $\{O\}$。

### 1. 接触运动学 (Contact Kinematics)
每个手指与物体表面的接触可以建模为不同的约束：
- **点接触 (Point Contact)**：仅传递三个维度的力，允许物体绕接触点转动。
- **带摩擦的点接触 (Point Contact with Friction, PCWF)**：最常用的模型，通过[摩擦锥](../formalizations/friction-cone.md)约束切向滑动。
- **软接触 (Soft Contact)**：允许传递法向扭矩。

### 2. 抓取雅可比矩阵 (Grasp Jacobian, G)
定义物体速度 $\mathcal{V}_o$ 与接触点处速度 $u$ 的映射关系：
$$ u = G^T \mathcal{V}_o $$
其中 $G$ 被称为**抓取矩阵 (Grasp Matrix)**。

### 3. 手指雅可比矩阵 (Hand Jacobian, J)
定义手指关节速度 $\dot{q}$ 与接触点速度 $u$ 的映射关系：
$$ u = J \dot{q} $$

## 闭链约束方程

为了使手指不脱离物体且不刺穿物体，必须满足一致性条件：
$$ J \dot{q} = G^T \mathcal{V}_o $$

这组方程揭示了灵巧操作的本质：
- **可操作性 (Manipulability)**：给定 $\dot{q}$，能否产生任意方向的 $\mathcal{V}_o$。
- **抓取稳定性**：是否存在一组关节力矩 $\tau$，能在不滑移的前提下抵消任意外部扰动。

## 关键挑战

1. **工作空间受限**：由于手指长度有限且存在闭链约束，灵巧手的手内可操作空间远小于其自由摆动空间。
2. **奇异位姿**：多指系统极易进入内部奇异位姿，导致物体瞬间失去某个自由度的约束。
3. **滚动接触 (Rolling Contact)**：在重定向任务中，手指与物体的接触点在持续移动，这使得雅可比矩阵 $G$ 和 $J$ 具有时变性。

## 关联页面
- [Allegro Hand 实体](../entities/allegro-hand.md)
- [Tactile Sensing (触觉感知)](./tactile-sensing.md)
- [手内重定向 (In-hand Reorientation)](../methods/in-hand-reorientation.md)
- [Friction Cone (摩擦锥) 形式化](../formalizations/friction-cone.md)
- [人形机器人并联关节解算](./humanoid-parallel-joint-kinematics.md) — 下肢闭链踝与力分配（与抓取闭链对照阅读）

## 参考来源
- Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. (多指抓取的经典圣经)
- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md)
