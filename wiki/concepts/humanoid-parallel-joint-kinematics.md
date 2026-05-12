---
type: concept
tags: [humanoid, kinematics, closed-chain, parallel-mechanism, ankle, simulation]
status: complete
updated: 2026-05-12
related:
  - ../entities/asimov-v1.md
  - ../entities/humanoid-robot.md
  - ./dexterous-kinematics.md
  - ./armature-modeling.md
  - ../entities/modern-robotics-book.md
sources:
  - ../../sources/notes/humanoid-parallel-joint-kinematics.md
summary: "人形上常见并联/闭链关节（如并联踝）在机构学上需闭链运动学与力分配；控制与仿真常暴露为等效串联关节，二者不可混为一谈。"
---

# 人形机器人并联关节解算（Parallel / Closed-Chain Joint Kinematics）

**并联关节解算**在这里指：当多个驱动分支通过刚性闭链耦合到同一末端（或同一等效自由度）时，在**机构空间**建立「驱动变量 ↔ 末端位姿/速度/力」映射，并处理**冗余与约束一致性**的一整套问题；人形产品中典型例子是 **并联踝（如 RSU 类构型）**，而不是「两个电池并联」之类的电气含义。

## 为什么重要

- **控制与规划消费的变量**往往是「踝 pitch / roll」等低维任务空间；**真实力矩路径**却经过多电机、多连杆的耦合。若忽略这一层，Sim2Real 会在**等效惯量、摩擦与背隙、足底力分配**上系统性失配。
- **仿真资产**（URDF / MJCF）里常见做法是暴露 **2 个 hinge 等效踝**，而不展开完整闭链；这在 RL、MPC 接口上很方便，但若要做 **机构尺度的辨识或硬件在环**，必须显式回到闭链模型或 CAD 约束。

## 核心结构：把「解算」拆成三层

1. **几何运动学（位置/速度）**  
   给定各主动关节（或等效驱动长度），求足端/中间连杆位姿为 **闭链正运动学**；给定末端目标求驱动为 **闭链逆运动学**，通常需数值迭代（牛顿–拉夫森）或机构相关的解析子问题。教材层系统叙述见 [*Modern Robotics* 第 7 章](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf)（亦见本库 [`Modern Robotics` 实体页](../entities/modern-robotics-book.md)）。

2. **静力学 / 力矩映射（力分配）**  
   当独立驱动数大于任务自由度时，同一足底力旋量对应**无穷多组**电机力矩解；实际系统由**刚度、摩擦、限位与电机热边界**共同「选中」一支解。这是并联腿与串联腿在 **WBC / QP 力控** 上的关键差异点之一。

3. **仿真与控制接口（建模折衷）**  
   公开人形 MJCF 常在每条腿只挂 **`ankle_pitch` + `ankle_roll`** 等电机铰链，用低维模型对接策略；**不**等价于已在模型里展开 RSU 的完整闭链雅可比。工程含义见 [Asimov v1](../entities/asimov-v1.md) 中「踝关节：RSU 机构学、公开 MJCF 与工程上的解算」一节。

## 与人形其他闭链问题的关系

- **灵巧手抓取**、**双臂共持刚体**同样形成闭链，约束方程写法相通，但接触与摩擦占主导；见 [Dexterous Kinematics](./dexterous-kinematics.md) 与 [双臂操作](../tasks/bimanual-manipulation.md)。
- **双电机经减速器驱动同一关节轴**时，反射惯量可按并联求和近似，见 [Armature Modeling](./armature-modeling.md)；这是**动力学参数**层面，与**闭链几何雅可比**互补而非替代。

## 常见误区或局限

- **误区：把策略输出的两个踝角当成机构上的两个独立电机角**。在真实并联踝里，它们往往经过**固定耦合**映射到多路驱动；标定与符号约定需单独文档化（公开代码里常见「弯膝时对踝 pitch 做几何补偿」一类提示，仍不等于完整闭链力雅可比）。
- **误区：在仿真里调大踝 PD 就能复现「硬件刚度」**。若未建模闭链柔度、背隙与分路摩擦，真机仍会出现足端高频模态与触地冲击差异。
- **局限：通用 URDF 一条串联树**难以原生表达任意闭链；实践中常见路径是**串联树 + 额外约束（equality / custom）**、多体软件中的闭链模块，或直接在机构 CAD 层导出专用模型。

## 与其他页面的关系

- 产品级 **RSU 并联踝**叙事与公开 MJCF 折衷：[Asimov v1](../entities/asimov-v1.md)。
- 闭链在操作方向的类比：[Dexterous Kinematics](./dexterous-kinematics.md)。
- 并联/双驱动对**惯量建模**的影响：[Armature Modeling](./armature-modeling.md)。
- 人形平台总览：[人形机器人](../entities/humanoid-robot.md)。

## 关联页面

- [Asimov v1](../entities/asimov-v1.md)
- [Dexterous Kinematics](./dexterous-kinematics.md)
- [Armature Modeling](./armature-modeling.md)
- [人形机器人](../entities/humanoid-robot.md)
- [Modern Robotics（教材实体）](../entities/modern-robotics-book.md)

## 推荐继续阅读

- [closed-chain-ik-js（Garrett Johnson）](https://github.com/gkjohnson/closed-chain-ik-js) — 闭链 IK 与约束求解的可读参考实现
- [A Framework for Optimal Ankle Design of Humanoid Robots（arXiv:2509.16469）](https://arxiv.org/abs/2509.16469) — 人形踝设计综述语境
- [How we built humanoid legs from the ground up in 100 days（Menlo）](https://menlo.ai/blog/humanoid-legs-100-days) — RSU 并联踝的产品与机构动机

## 参考来源

- [sources/notes/humanoid-parallel-joint-kinematics.md](../../sources/notes/humanoid-parallel-joint-kinematics.md) — 本主题资料索引与 ingest 归档
- [sources/papers/modern_robotics_textbook.md](../../sources/papers/modern_robotics_textbook.md) — *Modern Robotics* 教材元数据与第 7 章指针
