# 模块依赖关系图

本页的目标不是做花哨图，而是先把 `Robotics_Notebooks` 当前最重要的依赖关系讲清楚。

重点回答：
1. 当前项目主线依赖怎么串
2. 哪些模块是上游基础，哪些模块是中层桥梁，哪些模块是落地执行层
3. 从哪里读起更顺

## 依赖关系的阅读方式

在这个项目里，“依赖”不只是一种：

- **知识依赖**：先理解 A，才更容易理解 B
- **建模依赖**：B 的建模形式建立在 A 之上
- **控制依赖**：B 的执行效果依赖 A 提供的状态、模型或参考
- **工程依赖**：真实系统要跑起来，必须把多条链同时接上

所以不要把这页理解成唯一真理的 DAG，它更像当前阶段最值得优先打通的主线图。

## 当前核心主线依赖

这是目前 `Robotics_Notebooks` 最核心的一条主线：

```text
LIP / ZMP
  ↓
Centroidal Dynamics
  ↓
Trajectory Optimization / MPC
  ↓
TSID / WBC
  ↓
State Estimation / System Identification
  ↓
Sim2Real
```

这条主线的意义：
- 前段解决“怎么抽象和规划人形运动”
- 中段解决“怎么把规划变成可执行控制”
- 后段解决“真实机器人状态和模型到底靠不靠谱”
- 最后落到“仿真训练和真实部署怎么接起来”

## 分层理解

## 1. 上游基础层

```text
数学基础
  ↓
刚体运动 / 机器人动力学 / 数值优化 / 概率统计
```

虽然这些内容目前还没在仓库里系统展开，但它们是当前主线的上游基础。

### 关键基础依赖
- **线性代数 / 微积分 / ODE** → 动力学建模 / 最优控制 / 状态估计
- **概率统计** → 状态估计 / 观测建模 / 部分可观测问题
- **数值优化** → Trajectory Optimization / MPC / QP-WBC / TSID

## 2. 建模与控制主干层

### 2.1 平衡与简化动力学链

```text
LIP / ZMP
  ↓
Centroidal Dynamics
```

关系：
- [LIP / ZMP](../wiki/concepts/lip-zmp.md) 提供双足平衡与步态控制的经典入门模型
- [Centroidal Dynamics](../wiki/concepts/centroidal-dynamics.md) 提供更接近真实足式 / 人形系统的中层动力学描述

理解顺序建议：
1. 先看 LIP / ZMP
2. 再看 Centroidal Dynamics

### 2.2 优化与规划链

```text
Optimal Control (OCP)
  ↓
Trajectory Optimization
  ↓
MPC
```

关系：
- [Optimal Control (OCP)](../wiki/concepts/optimal-control.md) 给出统一问题形式
- [Trajectory Optimization](../wiki/methods/trajectory-optimization.md) 解决整段轨迹怎么设计
- [Model Predictive Control (MPC)](../wiki/methods/model-predictive-control.md) 解决在线滚动优化怎么做

这里最重要的理解不是“谁替代谁”，而是：
- OCP 是问题母体
- Trajectory Optimization 更偏整段动作设计
- MPC 更偏在线闭环修正

### 2.3 控制落地链

```text
Centroidal / MPC / trajectory reference
  ↓
TSID / WBC
  ↓
joint torque / acceleration / contact force execution
```

关系：
- [TSID](../wiki/concepts/tsid.md) 负责把任务空间和动力学约束变成低层可执行控制
- [Whole-Body Control](../wiki/concepts/whole-body-control.md) 是更大的控制框架语境

这条链回答的是：

> 上层算出来的参考，怎么真正落到机器人身体上。

## 3. 估计与模型可信度层

### 3.1 状态链

```text
传感器观测
  ↓
State Estimation
  ↓
控制器 / 规划器可用状态
```

关系：
- [State Estimation](../wiki/concepts/state-estimation.md) 解决“机器人当前真实状态是什么”

如果这层不稳：
- MPC 会预测错
- TSID / WBC 会控制错
- RL policy 的观测也会偏

### 3.2 模型链

```text
实验数据
  ↓
System Identification
  ↓
更可信的动力学 / 执行器 / 接触模型
```

关系：
- [System Identification](../wiki/concepts/system-identification.md) 解决“模型到底准不准”

如果这层不稳：
- 预测模型会偏
- 控制力矩会偏
- 仿真和真机会脱节

## 4. Sim2Real 桥接层

```text
State Estimation + System Identification + Domain Randomization
  ↓
Sim2Real
```

关系：
- [Sim2Real](../wiki/concepts/sim2real.md)
- [Domain Randomization](../wiki/concepts/domain-randomization.md)

它不是某个单一模块，而是桥接层。

它依赖：
- 状态能不能估准
- 模型能不能拟合到位
- 训练分布有没有覆盖真实扰动

所以 sim2real 在依赖图里应该放在后段，不是起点。

## 5. 学习控制分支

```text
Reinforcement Learning / Imitation Learning
  ↓
Locomotion / humanoid skill learning
  ↓
Sim2Real
```

相关页面：
- [Reinforcement Learning](../wiki/methods/reinforcement-learning.md)
- [Imitation Learning](../wiki/methods/imitation-learning.md)
- [WBC vs RL](../wiki/comparisons/wbc-vs-rl.md)
- [Locomotion](../wiki/tasks/locomotion.md)

当前理解重点：
- 学习控制不是独立岛屿
- 它需要和控制、估计、sim2real 一起看
- 在当前项目里，学习分支的最好落点是 humanoid locomotion

## 横向桥接关系

除了主链，还有几条很重要的横向桥：

### 桥 1：Trajectory Optimization ↔ MPC
- 前者偏离线整段设计
- 后者偏在线滚动修正
- 两者常常组合使用

### 桥 2：Centroidal Dynamics ↔ TSID / WBC
- 前者负责中层动力学规划
- 后者负责低层动力学一致执行

### 桥 3：State Estimation ↔ TSID / MPC / RL Policy
- 估计质量直接决定控制和策略输入质量

### 桥 4：System Identification ↔ MPC / Sim2Real
- 模型对不对，直接决定预测和迁移效果

## 当前最推荐阅读顺序

如果你的目标是“人形机器人运动控制主线”，推荐顺序：

1. [LIP / ZMP](../wiki/concepts/lip-zmp.md)
2. [Centroidal Dynamics](../wiki/concepts/centroidal-dynamics.md)
3. [Optimal Control (OCP)](../wiki/concepts/optimal-control.md)
4. [Trajectory Optimization](../wiki/methods/trajectory-optimization.md)
5. [Model Predictive Control (MPC)](../wiki/methods/model-predictive-control.md)
6. [TSID](../wiki/concepts/tsid.md)
7. [Whole-Body Control](../wiki/concepts/whole-body-control.md)
8. [State Estimation](../wiki/concepts/state-estimation.md)
9. [System Identification](../wiki/concepts/system-identification.md)
10. [Sim2Real](../wiki/concepts/sim2real.md)
11. [Locomotion](../wiki/tasks/locomotion.md)

如果你的目标是“学习控制切入”，推荐顺序：

1. [Robot Learning Overview](../wiki/overview/robot-learning-overview.md)
2. [Reinforcement Learning](../wiki/methods/reinforcement-learning.md)
3. [Imitation Learning](../wiki/methods/imitation-learning.md)
4. [Locomotion](../wiki/tasks/locomotion.md)
5. [WBC vs RL](../wiki/comparisons/wbc-vs-rl.md)
6. [State Estimation](../wiki/concepts/state-estimation.md)
7. [System Identification](../wiki/concepts/system-identification.md)
8. [Sim2Real](../wiki/concepts/sim2real.md)

## 当前结论

这页后续不是去追求漂亮图，而是持续维护三件事：

1. 当前主线依赖有没有讲清楚
2. 新增页面有没有挂回依赖图
3. 入口顺序和学习顺序有没有越来越顺

也就是说：

> `tech-map/dependency-graph.md` 的职责，是把“项目里有哪些页面”升级成“这些页面之间为什么有先后关系、为什么值得按这条链读”。
