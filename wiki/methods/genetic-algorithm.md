---
type: method
tags: [evolutionary-computation, genetic-algorithm, optimization, robotics, locomotion]
status: complete
updated: 2026-10-10
related:
  - ./genetic-programming.md
  - ./reinforcement-learning.md
  - ../concepts/reinforcement-learning-history.md
  - ../tasks/locomotion.md
  - ../entities/paper-ga-biped-slope-gait.md
sources:
  - ../../sources/papers/genetic_algorithms_foundations.md
  - ../../sources/papers/evolutionary_optimization_foundations.md
summary: "遗传算法（GA）以群体适应度、选择、交叉和变异搜索候选解；适合不可微、非凸或仿真可评估的问题，但不保证全局最优，也不是强化学习。"
---

# 遗传算法（Genetic Algorithm, GA）

## 一句话定义

**遗传算法**是一种群体式黑箱优化：将候选解编码成染色体，以适应度评价后反复选择、交叉和变异，逐代寻找更好的候选。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GA | Genetic Algorithm | 用编码、适应度和遗传算子迭代搜索 |
| EA | Evolutionary Algorithm | 以群体、选择和变异为核心的一类优化方法 |
| GP | Genetic Programming | 演化可执行程序或表达式结构，不只是参数串 |
| ES | Evolution Strategy | 常用于连续优化、通过变异搜索并自适应搜索尺度 |
| CMA-ES | Covariance Matrix Adaptation Evolution Strategy | 学习高斯搜索分布协方差和步长的优化算法 |
| DE | Differential Evolution | 利用种群个体差分产生候选向量的算法 |
| ZMP | Zero Moment Point | 双足步态案例中的稳定性/支撑可行性指标 |

## 为什么重要

机器人优化常面对非凸目标、离散选择、接触仿真和不可微评价。GA 不需要梯度，只需要一致地评价候选方案，可用于轨迹系数、落脚参数、控制器配置和机构设计的外层搜索。其结果由编码、适应度和评估预算共同决定，“进化”本身不会自动发现好解。

## 主要技术路线

按决策变量与表示选搜索家族：离散/组合编码优先看经典 GA；连续且变量相关的黑箱参数可比较 CMA-ES；连续目标也可将 DE 作为简单基线。机器人落地常把 GA 放在轨迹或控制器参数的离线外层，而非替代高频闭环控制。

## 核心原理与流程

一个个体是候选解的编码，种群是一批候选解，适应度将任务表现转为排序或选择信号。典型一代循环如下：

```mermaid
flowchart LR
  init["初始化候选种群"] --> eval["仿真评估适应度"]
  eval --> select["按适应度选择父代"]
  select --> vary["交叉与变异"]
  vary --> next["形成下一代"]
  next --> check{"满足终止条件？"}
  check -->|否| eval
  check -->|是| best["输出验证后的候选解"]
```

1. **编码：**可用二进制串、实数向量、排列或专用结构；编码应让遗传操作尽量生成合法方案。
2. **评估：**在固定任务分布下计算适应度；多目标机器人任务需记录成功、稳定、能耗、冲击和执行器约束。
3. **选择：**锦标赛、比例或排序选择偏向高适应度个体；选择压力太大易早熟。
4. **交叉与变异：**交叉重组父代，变异注入新信息；精英保留可避免最佳已知解丢失，但不保证全局最优。
5. **停止与验证：**按预算、若干代无改进或阈值停止；用独立种子、未见地形复测，避免选择到仿真噪声赢家。

## 机器人里的用法

高频闭环人形控制通常由梯度 RL 等策略学习承担；GA 更适合外层低频配置或参数搜索：

- **步态/轨迹参数：**搜索摆腿、步长、周期和质心轨迹系数。仓库案例 [GA 坡面双足](../entities/paper-ga-biped-slope-gait.md)在 8-DoF 模型上以 GA + ZMP 惩罚优化轨迹，报告仿真结果。
- **控制器参数：**离线搜索 PD/阻抗、滤波或任务权重，同时约束稳定、力矩/电流及温升。
- **跨场景评估：**跟踪误差、行走距离、摔倒、能耗与冲击可组成适应度；训练种子和验证种子分离。

### 相邻算法如何选

| 方法 | 搜索对象 / 信号 | 优先考虑 |
|------|-----------------|----------|
| GA | 显式编码的参数、排列或组合解 | 离散结构、组合约束或自定义表示 |
| [GP](./genetic-programming.md) | 可执行程序树/表达式 | 发现公式、规则或程序结构 |
| CMA-ES | 连续向量上的高斯协方差/步长 | 连续、变量相关的黑箱参数 |
| DE | 连续向量差分变异 + 选择 | 连续黑箱目标的简洁基线 |

GA 是进化算法的一员；它与 [强化学习](./reinforcement-learning.md) 的差别不在是否出现“reward/fitness”字样：RL 学习状态条件策略并利用序贯状态转移，GA 通常评估整条候选方案后做群体搜索。与 GP 的差别是染色体表示固定参数/结构，还是可变程序。

## 工程实践

- 先选低维、可解释参数化；把大规模策略权重直接塞进基础 GA 往往评估成本过高。
- 记录原始多指标和约束违反，再明确适应度归一化/权重；避免只看综合分。
- 每代监控最优值、中位数、多样性和模拟器失败率，识别早熟或通过仿真漏洞刷分。
- 固定仿真版本、控制频率和终止条件；最终候选先跨种子、地形复核，再进入硬件在环和真机小步验证。

## 局限与风险

- 机器人 rollout 可能很贵，评估预算直接限制种群规模与代数。
- 编码与算子偏置可达解；坏编码会令有效方案不可表达或产生大量不可行后代。
- 选择压力和噪声可造成早熟收敛或对仿真噪声过拟合。
- 普通 GA 不保证有限预算找到全局最优；搜索结果应称为当前预算下的候选。

## 关联页面

- [遗传编程（GP）](./genetic-programming.md)
- [强化学习史](../concepts/reinforcement-learning-history.md)
- [Locomotion](../tasks/locomotion.md)
- [GA 坡面双足](../entities/paper-ga-biped-slope-gait.md)

## 参考来源

- [Holland / Goldberg 遗传算法原著归档](../../sources/papers/genetic_algorithms_foundations.md)
- [CMA-ES / DE 原始论文归档](../../sources/papers/evolutionary_optimization_foundations.md)
- Holland, *Adaptation in Natural and Artificial Systems*, [MIT Press](https://mitpress.mit.edu/9780262082136/adaptation-in-natural-and-artificial-systems/).
- Goldberg & Holland, [“Genetic Algorithms and Machine Learning”](https://doi.org/10.1007/BF00113892), 1988.

## 推荐继续阅读

- [Holland 原著（MIT Press）](https://mitpress.mit.edu/9780262082136/adaptation-in-natural-and-artificial-systems/)
- [Goldberg & Holland 1988（Springer）](https://doi.org/10.1007/BF00113892)
- [差分进化原论文（Springer）](https://doi.org/10.1023/A:1008202821328)
