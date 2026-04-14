# Contact Dynamics

**Contact Dynamics（接触动力学）**：研究机器人与地面、物体、墙面等环境发生接触时，接触力、约束和系统运动之间关系的动力学框架。

## 一句话定义

如果说 Floating Base Dynamics 研究的是“机器人底座不固定时整机怎么动”，那 **Contact Dynamics** 研究的就是：

> 机器人一旦和环境发生接触，这些接触力和接触约束会怎样改变系统动力学。

一句话说白了：

> 对腿式机器人来说，接触不是场景细节，而是系统动力学本身的一部分。

## 为什么重要

很多机器人问题一旦进入真实世界，核心难点马上从“自由空间运动”变成“接触”。

比如：
- 人形机器人站立
- 双足机器人迈步
- 四足机器人跑动
- 机械臂抓取
- 全身操作时身体支撑墙面或桌面

这些问题都离不开一个事实：

> **一旦有接触，运动不再只由关节驱动决定，而是同时由接触约束和接触力共同决定。**

所以 Contact Dynamics 重要的原因是：
- 它解释为什么腿式机器人能站住
- 它解释为什么接触切换会让控制难度暴涨
- 它解释为什么 WBC / TSID / MPC 里接触力是核心变量
- 它解释为什么 locomotion 本质上不是单纯轨迹跟踪，而是接触组织问题

## 什么叫接触

在机器人里，“接触”不只是碰到了某个物体。

更准确地说，接触意味着：
- 某些自由度被环境约束
- 系统可施加 / 承受的力发生改变
- 运动学和动力学方程都要变

### 常见接触形式
- 足底接触地面
- 机械臂末端接触物体
- 手或肘支撑环境
- 多点接触（双脚 + 手）
- 接触建立与断开（impact / lift-off）

## 接触为什么让问题突然变难

### 1. 运动不再是自由运动
如果机器人完全在空中，它的运动只受自身动力学和控制输入影响。

一旦接触建立：
- 某些点不应该再乱动
- 系统开始受到外部反作用力
- base motion 会被接触力强烈影响

### 2. 接触力不是你随便指定就行
接触力必须满足：
- 法向力不能把地面“拉起来”
- 摩擦锥约束要满足，不然会打滑
- 接触点速度 / 加速度约束要满足

也就是说，接触力不是任意变量，而是受物理约束严格限制的。

### 3. 接触切换带来不连续性
最难的地方之一是：
- 脚抬起
- 脚落地
- 双脚切成单脚支撑
- 冲击发生

这些时刻往往会让动力学、约束和控制目标一起变化。

## 接触如何进入动力学方程

在 floating base 系统里，经典动力学通常写成：

$$
M(q) \ddot{q} + h(q, \dot{q}) = S^T \tau + J_c^T f
$$

这里接触通过两部分进入：

### 1. 接触力项

$$
J_c^T f
$$

表示环境通过接触对机器人施加广义力。

其中：
- \( J_c \) 是接触雅可比
- \( f \) 是接触力

这部分告诉你：

> 地面反力、墙面反力、抓取接触力，都会直接改写机器人动力学。

### 2. 接触约束项

当某个接触点固定在环境上时，通常有：

$$
J_c \ddot{q} + \dot{J}_c \dot{q} = 0
$$

意思是：
- 接触点不能乱加速
- 它的运动要满足接触一致性

这相当于给系统额外施加了运动学 / 动力学约束。

## 最重要的几个接触概念

### 1. Normal Force（法向力）
地面只能“推”机器人，通常不能“拉”机器人。

所以常见约束是：

$$
f_n \ge 0
$$

### 2. Friction Cone（摩擦锥）
如果切向力太大，就会打滑。

所以接触力必须满足摩擦锥约束：

$$
\|f_t\| \le \mu f_n
$$

这里：
- \( f_t \) 是切向力
- \( f_n \) 是法向力
- \( \mu \) 是摩擦系数

### 3. Support Polygon（支撑多边形）
在腿式机器人里，接触区域决定稳定余地。

### 4. Contact Schedule（接触时序）
什么时候哪只脚接触地面，什么时候抬起，是 locomotion 的核心变量之一。

## 和 Floating Base Dynamics 的关系

Floating Base Dynamics 解释了：
- base 不固定
- 整机动力学怎么组织

Contact Dynamics 进一步解释：
- 接触力怎样改变这套动力学
- 接触约束怎样改变系统运动可能性

所以关系是：

```text
Floating Base Dynamics
    ↓
Contact Dynamics
    ↓
Centroidal Dynamics / TSID / WBC / Locomotion
```

见：[Floating Base Dynamics](./floating-base-dynamics.md)

## 和 Centroidal Dynamics 的关系

Centroidal Dynamics 非常依赖接触力，因为：
- 线动量变化由接触力决定
- 角动量变化也由接触力和力臂决定

很多 centroidal planner 本质上就是：
- 优化接触力
- 满足摩擦锥与支撑约束
- 让质心和动量按目标演化

见：[Centroidal Dynamics](./centroidal-dynamics.md)

## 和 TSID / WBC 的关系

TSID / WBC 常常在做：
- 接触力分配
- 接触一致性约束
- 末端任务和接触约束的协调

所以接触动力学不是它们的附属知识，而是核心前置。

见：[TSID](./tsid.md)

见：[Whole-Body Control](./whole-body-control.md)

## 和 Locomotion 的关系

locomotion 本质上可以看成是：

> 在不断变化的接触时序下，组织平衡、动量、足端和接触力的一系列动作。

所以接触动力学几乎就是 locomotion 的骨架之一。

见：[Locomotion](../tasks/locomotion.md)

## 和 State Estimation 的关系

接触不只是控制问题，也是估计问题：
- 脚到底有没有接触地面
- 接触点是不是在滑
- 当前支撑状态是什么

这些都直接影响：
- base velocity estimation
- gait phase 判断
- 控制器是否信任接触约束

见：[State Estimation](./state-estimation.md)

## 在机器人中的典型应用

### 1. 人形站立与行走
双足机器人几乎所有稳定控制都离不开接触力与支撑约束。

### 2. 四足跑动
四足 locomotion 对接触切换、摩擦约束和地面适应非常敏感。

### 3. 抓取与操作
机械臂抓住物体之后，系统动力学会因为接触而改变。

### 4. 全身操作
身体、手、脚同时接触环境时，接触动力学复杂度会明显上升。

## 常见误区

### 1. 以为接触只是“碰撞检测”
不止。碰撞检测只是接触建立的一部分，真正难的是接触后的力和约束怎么进系统。

### 2. 以为接触力是控制器想给多少就给多少
不行，必须满足法向力、摩擦锥、接触一致性等物理约束。

### 3. 以为接触问题只影响足式机器人
机械臂抓取、全身操作、支撑动作同样强依赖接触动力学。

### 4. 低估接触切换的难度
脚抬起和落地那一瞬间，往往是最容易炸控制器的地方。

## 推荐使用建议

### 如果你做人形 / 四足
这页是必须懂的，因为 locomotion 的本质就是接触组织问题。

### 如果你做 WBC / TSID / MPC
必须理解接触如何进入动力学方程，不然会一直觉得“接触力”只是某种优化变量，而不是物理核心量。

### 如果你做 RL locomotion
即使策略是端到端学的，也建议懂接触动力学，不然 reward、observation、termination 条件很容易设计得很盲。

## 参考来源

- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — ingest 档案（Featherstone 2008 / Stewart LCP 2000 / Todorov MuJoCo 2012）
- Featherstone, *Rigid Body Dynamics Algorithms* — 刚体接触动力学核心算法参考
- Murray et al., *A Mathematical Introduction to Robotic Manipulation* — 接触约束与闭链建模
- Mirtich & Canny, *Impulse-based Simulation of Rigid Bodies* (1995) — 冲量接触建模基础

## 推荐继续阅读

- Featherstone, *Rigid Body Dynamics Algorithms*
- [Floating Base Dynamics](./floating-base-dynamics.md)
- [Centroidal Dynamics](./centroidal-dynamics.md)
- [TSID](./tsid.md)
- [Contact Complementarity](../formalizations/contact-complementarity.md) — 接触动力学的 LCP 数学框架

## 一句话记忆

> Contact Dynamics 研究的是“机器人一旦和环境发生接触，接触力和接触约束如何改写系统动力学”，它是 locomotion、WBC、TSID、state estimation 和 sim2real 的共同前置。
