---
type: entity
tags: ["paper", "wbc", "operational-space", "force-control", "classic", "hmi-papers"]
status: complete
updated: 2026-07-31
venue: "HMI curated · 1987"
summary: "OSF / Operational Space Formulation（HMI P001）：把运动与力控制直接写在末端任务空间动力学上，再用动态一致映射得到关节力矩，为后续任务优先级与全身控制奠基。"
related:
  - ../concepts/whole-body-control.md
  - ../concepts/hybrid-force-position-control.md
  - ../concepts/hqp.md
  - ./paper-hmi-stack-of-tasks.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p001_operational-space-formulation.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# OSF / Operational Space Formulation（HMI P001）

**OSF / Operational Space Formulation**（*A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation*，1987）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P001**，主分类为 **工程与实机部署**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

把运动与力控制直接写在末端任务空间动力学上，再用动态一致映射得到关节力矩，为后续任务优先级与全身控制奠基。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OSF | Operational Space Formulation | 在任务/操作空间统一表达运动与力控制 |
| WBC | Whole-Body Control | 全身多任务控制，继承操作空间思路 |
| IK | Inverse Kinematics | 由任务空间目标求关节运动 |
| PD | Proportional–Derivative | 常见低层关节/任务跟踪器 |

## 为什么重要

- 机器人关节动力学给出质量矩阵、科氏/离心项、重力和关节力矩之间的关系，末端速度则由雅可比把关节速度映射到任务空间。关键不是简单使用雅可比转置，而是把关节空间惯量也一起映射过去，得到任务空间等效惯量 `Lambda`。这样，期望的任务空间加速度或力可以经过动态一致的映射变成关节力矩；控制器知道机器人沿不同方向“有多重”，而不是把所有方向当成同一种运动学误差。
- 在 HMI 六条技术路线中属于 **工程与实机部署**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P001 |
| 年份 | 1987 |
| 分组 | 工程与实机部署 |
| 开源状态 | 不适用（经典论文，无可运行现代训练仓） |
| 原文 | https://doi.org/10.1109/JRA.1987.1087068 |

## 核心原理

这篇论文真正改变的不是一个控制增益，而是控制问题的表达方式。传统关节控制先给每个关节一个目标，再希望末端得到想要的运动；Khatib反过来问：如果任务本来就是“手沿某方向运动并在另一方向施力”，为什么不直接在末端任务空间定义动力学，再把结果映射成关节力矩？后来的操作空间控制、任务优先级和Whole-Body Control都继承了这个出发点。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["OSF / Operational Space Formulation"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

冗余自由度也因此有了明确去处：主任务占用的方向由任务控制，剩余自由度通过动态一致零空间完成姿态、避限位等次任务，并尽量不扰动主任务。这里的“动态一致”很重要，普通伪逆只保证速度层面的投影，未必保证施加次任务力矩后主任务加速度不受影响。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 不适用（经典论文，无可运行现代训练仓） |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（不适用（经典论文，无可运行现代训练仓））。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**OSF / Operational Space Formulation 应作为 HMI「工程与实机部署」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：不适用（经典论文，无可运行现代训练仓）。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 接触任务通常不是所有方向都做位置控制。例如末端沿平面切向移动，同时在法向维持接触力。论文用选择矩阵把任务空间分成互补的运动方向和力方向，两部分最后都通过同一套任务空间动力学进入关节力矩。它并不是让位置环和力环互相“抢控制权”，而是先规定各自负责哪些方向。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [hybrid-force-position-control](../concepts/hybrid-force-position-control.md)
- [hqp](../concepts/hqp.md)
- [paper-hmi-stack-of-tasks](./paper-hmi-stack-of-tasks.md)

## 参考来源

- [sources/papers/hmi_p001_operational-space-formulation.md](../../sources/papers/hmi_p001_operational-space-formulation.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [DOI](https://doi.org/10.1109/JRA.1987.1087068)
- [HMI 逐篇解读 P001](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P001.md)
