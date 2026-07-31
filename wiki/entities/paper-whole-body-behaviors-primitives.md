---
type: entity
tags: ["paper", "wbc", "hierarchical-control", "classic", "hmi-papers"]
status: complete
updated: 2026-07-31
venue: "HMI curated · 2005"
summary: "Whole-Body Behaviors（HMI P002）：把全身行为拆成有严格优先级的行为基元：高优先级先占用自由度，低优先级只在动态一致零空间内工作。"
related:
  - ../concepts/whole-body-control.md
  - ./paper-operational-space-formulation.md
  - ./paper-hmi-stack-of-tasks.md
  - ../concepts/hqp.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p002_whole-body-behaviors-primitives.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Whole-Body Behaviors（HMI P002）

**Whole-Body Behaviors**（*Synthesis of Whole-Body Behaviors through Hierarchical Control of Behavioral Primitives*，2005）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P002**，主分类为 **工程与实机部署**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

把全身行为拆成有严格优先级的行为基元：高优先级先占用自由度，低优先级只在动态一致零空间内工作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 全身行为合成与多任务控制 |
| HQP | Hierarchical Quadratic Programming | 后来常见的层级二次规划求解形态 |
| CoM | Center of Mass | 平衡任务常用质心目标 |
| OSF | Operational Space Formulation | 任务空间动力学表述前驱 |

## 为什么重要

- 任务层在操作空间中描述手、质心或躯干等目标，利用任务空间动力学计算相应控制力；姿态层处理冗余关节，希望机器人在完成任务的同时维持自然或可控的全身构型。每增加一个低优先级任务，都要先投影到前面所有高优先级任务的动态一致零空间。于是“拿箱子”不是手、质心和躯干三个控制器简单叠加，而是一个有明确支配关系的递归结构。
- 在 HMI 六条技术路线中属于 **工程与实机部署**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P002 |
| 年份 | 2005 |
| 分组 | 工程与实机部署 |
| 开源状态 | 不适用（经典论文） |
| 原文 | https://doi.org/10.1142/S0219843605000594 |

## 核心原理

让人形机器人双手拿箱子时，手要到位，质心不能跑出支撑区域，躯干还要保持合适姿态。最直接的做法是给所有误差加权求和，但权重稍有变化，平衡任务就可能被手部任务牺牲。这篇论文的核心思想是把行为拆成有严格优先级的primitive：高优先级任务先占用它需要的自由度，低优先级任务只能在不破坏前者的剩余空间里工作。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["Whole-Body Behaviors"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

这种做法的价值在冲突时最明显：如果双手目标和保持平衡不能同时满足，系统应先保住平衡，再在剩余能力内尽量靠近手部目标。严格层级比加权和更容易表达这种安全语义，但也意味着优先级选错会造成低层任务长期没有自由度，任务切换还可能带来不连续。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 不适用（经典论文） |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（不适用（经典论文））。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**Whole-Body Behaviors 应作为 HMI「工程与实机部署」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：不适用（经典论文）。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 控制输入由机器人状态、任务参考和当前接触组成，每个primitive输出任务误差、雅可比及期望任务力；递归投影后合成为关节力矩。高层行为只需激活、停用或更新目标，底层层级负责同一控制周期内的资源分配。这个接口后来可以由规划器、遥操作或学习策略提供任务参考，但平衡优先级不能交给上层任意覆盖。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [paper-operational-space-formulation](./paper-operational-space-formulation.md)
- [paper-hmi-stack-of-tasks](./paper-hmi-stack-of-tasks.md)
- [hqp](../concepts/hqp.md)

## 参考来源

- [sources/papers/hmi_p002_whole-body-behaviors-primitives.md](../../sources/papers/hmi_p002_whole-body-behaviors-primitives.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [DOI](https://doi.org/10.1142/S0219843605000594)
- [HMI 逐篇解读 P002](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P002.md)
