---
type: concept
tags: [dexterity, kinematics, robot-hand, math, manipulation]
status: complete
updated: 2026-09-15
related:
  - ../queries/contact-wrench-closed-loop.md
  - ./humanoid-parallel-joint-kinematics.md
  - ./tactile-sensing.md
  - ../entities/allegro-hand.md
  - ../entities/all-hands-up.md
  - ../entities/paper-digitcode.md
  - ../entities/paper-mango-grasp.md
  - ../methods/in-hand-reorientation.md
  - ../formalizations/friction-cone.md
sources:
  - ../../sources/papers/contact_dynamics.md
  - ../../sources/papers/in_hand_manipulation_survey_reading_arxiv_2401_07915_dk1.md
original:
  - ../../sources/papers/in_hand_manipulation_survey_reading_arxiv_2401_07915_dk1.md
summary: "灵巧手运动学（Dexterous Kinematics）研究多指末端执行器与被操作物体之间的运动映射关系，其核心在于处理多点接触形成的闭链约束。"
---

# Dexterous Kinematics (灵巧手运动学)

**灵巧手运动学 (Dexterous Kinematics)** 是机器人学中研究多指协同操作的理论基础。与传统的单臂串联运动学不同，灵巧手在抓取物体时，多个手指通过接触点与物体共同构成了一个**闭链运动学系统**。

## 一句话定义

灵巧手运动学研究多指与物体经接触形成闭链后，关节速度、接触点速度与物体速度之间的映射（$J\dot q = G^T\mathcal V_o$），是判断手内可操作性与抓取稳定性的几何前提。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| PCWF | Point Contact with Friction | 带摩擦的点接触模型 |
| DOF | Degrees of Freedom | 自由度 |

## 为什么重要

手内重定向、指步态等灵巧操作都建立在闭链约束之上：接触点怎么动、哪些方向被约束，决定了策略能否稳定操作物体。解析运动学给出可操作性与奇异性的判据，学习方法（RL、模仿）则常把它作为隐含结构或仿真前提。

## 核心原理

见下文「核心数学模型」与「闭链约束方程」：接触模型、抓取矩阵 $G$、手指雅可比 $J$ 及其一致性条件。

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

## 工程实践

据 Weinberg 等综述（arXiv 2401.07915，已通读）：
- 刚性手的前向运动学已知，转移模型可解析得到；柔顺 / 欠驱动 / 软手难以精确解析建模，常改用数据驱动的转移模型。
- 手内操作按是否保持连续接触分为滚动、枕转、滑动、手内（in-grasp）、指步态等；保持连续接触时须全程保证抓取稳定，指步态通常准静态执行。
- 综述引 Mason 与 Salisbury：刚性手至少 3 指、每指 3 关节才可对物体获得可控性。
- 仿真（MuJoCo、Isaac Gym）里接触力不够可靠，有工作只用二值触觉信号以便 sim-to-real；域随机化是常见手段。

## 局限

- 上文公式假设接触点已知且保持，滚动 / 滑动接触使 $G$、$J$ 时变（见「关键挑战」）。
- 综述指出解析模型难以应对复杂环境与不确定性，摩擦、软体形变与接触力在仿真中仍是 sim-to-real 缺口。
- 本页的 $G$、$J$ 推导取自教材 Murray–Li–Sastry，该书未在本次通读范围内；综述通读只支撑上述工程与局限部分。

## 关联页面
- [Query：接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md) — 灵巧手接触点雅可比是把 6D 力旋量映回关节的运动学前提
- [Allegro Hand 实体](../entities/allegro-hand.md)
- [Tactile Sensing (触觉感知)](./tactile-sensing.md)
- [手内重定向 (In-hand Reorientation)](../methods/in-hand-reorientation.md)
- [Friction Cone (摩擦锥) 形式化](../formalizations/friction-cone.md)
- [人形机器人并联关节解算](./humanoid-parallel-joint-kinematics.md) — 下肢闭链踝与力分配（与抓取闭链对照阅读）
- [DigitCode](../entities/paper-digitcode.md) — 按解剖单元符号化手姿态
- [MANGO-Grasp](../entities/paper-mango-grasp.md) — 形态–运动学描述子 + 马氏接触场
- [All Hands Up](../entities/all-hands-up.md) — 用仿真 Kapandji 把拇指对掌工作空间收成可对照分数

## 参考来源
- Weinberg, A. I., Shirizly, A., Azulay, O., & Sintov, A. *Survey of Learning-based Approaches for Robotic In-Hand Manipulation*. arXiv:2401.07915. 归档：[sources/papers/in_hand_manipulation_survey_reading_arxiv_2401_07915_dk1.md](../../sources/papers/in_hand_manipulation_survey_reading_arxiv_2401_07915_dk1.md)
- Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. (多指抓取的经典圣经)
- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md)

## 继续阅读
- [手内重定向 (In-hand Reorientation)](../methods/in-hand-reorientation.md)
- [Tactile Sensing (触觉感知)](./tactile-sensing.md)
