---
type: method
tags: [evolutionary-computation, genetic-programming, program-synthesis, symbolic-regression, robotics]
status: complete
updated: 2026-10-10
related:
  - ./genetic-algorithm.md
  - ./reinforcement-learning.md
  - ../concepts/reinforcement-learning-history.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/genetic_programming_foundations.md
  - ../../sources/papers/evolutionary_optimization_foundations.md
summary: "遗传编程（GP）把程序/表达式本身当作个体，在函数与终端集合定义的语法空间中按适应度演化；它搜索程序结构，而不只是优化既定策略参数。"
---

# 遗传编程（Genetic Programming, GP）

## 一句话定义

**遗传编程**是将程序表示为可执行个体并通过适应度、选择、交叉和变异逐代演化的自动程序搜索方法；搜索对象是程序结构，而不只是固定维度参数。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GP | Genetic Programming | 在程序/表达式空间中演化可执行解 |
| GA | Genetic Algorithm | 以编码染色体做群体搜索，可只优化参数 |
| ADF | Automatically Defined Function | GP 个体可演化和复用的子程序模块 |
| AST | Abstract Syntax Tree | 用树结构表达程序语法和运算层级 |
| RL | Reinforcement Learning | 依据序贯状态—动作反馈学习策略，与整体程序适应度搜索不同 |

## 为什么重要

GP 能在预先限定的函数、变量和常量集合中搜索规则、公式或控制程序。当任务可由仿真重复评估、但解析梯度难获得时，可探索紧凑控制律或符号模型。搜索边界由人定义，所以“自动编程”不等于无需领域建模。

## 主要技术路线

- **树型 GP：**按函数/终端语法演化表达式树，是 Koza 原著的经典表示。
- **模块化 GP：**借助 ADF 演化可复用子程序，适合出现重复结构的候选程序。
- **约束语法 / 有类型 GP：**将领域规则、输入输出类型或单位嵌入生成和变异过程，缩小非法程序搜索空间；机器人控制尤其应限制危险算子。

## 核心原理与流程

GP 个体常是一棵程序树：内部节点来自函数集合（如运算、比较、条件或控制原语），叶节点来自终端集合（观测、常量、传感器状态）。执行后输出动作或预测，再由适应度评分。

```mermaid
flowchart LR
  grammar["定义函数集与终端集"] --> init["生成程序树种群"]
  init --> run["在任务/仿真中执行"]
  run --> fit["计算适应度与约束"]
  fit --> select["选择父代程序"]
  select --> edit["子树交叉与变异"]
  edit --> next["替换/保留个体"]
  next --> check{"满足终止条件？"}
  check -->|否| run
  check -->|是| verify["复测并检查程序"]
```

1. **定义语法与类型：**限制变量、常量、操作符、单位和返回类型；机器人控制宜用有类型算子，避免角度、位置、力矩等不兼容运算。
2. **初始化：**用受限深度和多样化策略生成合法树，防止初代个体过深。
3. **执行与评估：**所有候选在相同任务条件下 rollout；评估任务误差、稳定、能耗、平滑、约束和程序复杂度。
4. **选择与改写：**按适应度选父代，子树交叉组合片段，变异替换子树、算子或常量。
5. **复测与隔离：**用未见扰动种子验证，检查数值边界与最坏运行时间；沙盒执行，安全限幅后才接近硬件。

### ADF 与模块化 GP

Koza 的 ADF 让系统在演化过程中产生可复用子程序和主程序，类似将重复计算抽成函数；它能增加结构复用，也会增加参数传递和搜索耦合。

## 与 GA、RL 的边界

| 维度 | GA | GP | RL |
|------|----|----|----|
| 被优化对象 | 固定表示的参数、排列或组合解 | 语法限定的程序树/表达式 | 状态条件策略及其参数 |
| 典型反馈 | 候选整体适应度 | 程序运行后的适应度 | 每步/轨迹回报与状态转移 |
| 结构是否变化 | 通常固定编码结构 | 程序树大小和结构可变 | 网络结构通常预先给定 |
| 常见用途 | 参数、轨迹、结构外层搜索 | 公式、规则、程序/控制律合成 | 高维连续控制与序贯决策 |

GP 与 [GA](./genetic-algorithm.md) 同属进化计算，却不是同义词；若要学习高频闭环 locomotion，RL 往往更自然；GP 更适合做受限的离线规则搜索。

## 工程实践

- 函数集/终端集就是归纳偏置：只允许量纲合理、数值稳定的算子，显式处理除零、溢出和角度周期性。
- 分层筛选：先静态语法/类型检查，再快速仿真，最后对少量候选做多种子高精度评测和硬件在环。
- 同时评价任务性能和树复杂度，防止 bloat 让程序变长却没有性能收益。
- 输出限制在安全动作区间；仿真高分不等于真机安全。

## 局限与风险

- 程序搜索空间组合爆炸，且执行每个候选也有成本。
- 树膨胀会拖慢评估、增加部署延迟并降低可读性。
- 结果依赖人工给定的语法、变量与适应度；GP 不会自动发现问题定义。
- 仿真漏洞、数值异常或 reward hacking 可能产生危险规则，须做约束评估和独立复测。

## 关联页面

- [遗传算法（GA）](./genetic-algorithm.md)
- [强化学习史](../concepts/reinforcement-learning-history.md)
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- [Koza GP 原著与 ADF 论文](../../sources/papers/genetic_programming_foundations.md)
- Koza, [*Genetic Programming: On the Programming of Computers by Means of Natural Selection*](https://mitpress.mit.edu/9780262111706/genetic-programming/), MIT Press, 1992.
- Koza, [“Genetic programming as a means for programming computers by natural selection”](https://doi.org/10.1007/BF00175355), 1994.
- [CMA-ES 与差分进化原始论文](../../sources/papers/evolutionary_optimization_foundations.md)

## 推荐继续阅读

- [Koza 1992 原著（MIT Press）](https://mitpress.mit.edu/9780262111706/genetic-programming/)
- [Koza 1994 ADF 论文（Springer）](https://doi.org/10.1007/BF00175355)
