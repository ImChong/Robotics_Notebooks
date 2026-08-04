---
type: entity
tags: ["paper", "wbc", "inverse-kinematics", "stack-of-tasks", "hmi-papers"]
status: complete
updated: 2026-07-31
code: https://github.com/stack-of-tasks/sot-core
venue: "HMI curated · 2009"
summary: "Stack of Tasks（HMI P003）：把任务、约束、雅可比与求解器组织成可动态插拔的软件任务栈，用零空间递归完成人形广义逆运动学。"
related:
  - ../concepts/hqp.md
  - ../concepts/tsid.md
  - ../concepts/whole-body-control.md
  - ./paper-operational-space-formulation.md
  - ./crocoddyl.md
sources:
  - ../../sources/papers/hmi_p003_hmi-stack-of-tasks.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Stack of Tasks（HMI P003）

**Stack of Tasks**（*A Versatile Generalized Inverted Kinematics Implementation for Collaborative Working Humanoid Robots: The Stack of Tasks*，2009）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P003**，主分类为 **工程与实机部署**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

把任务、约束、雅可比与求解器组织成可动态插拔的软件任务栈，用零空间递归完成人形广义逆运动学。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SoT | Stack of Tasks | 可动态组合的层级任务栈框架 |
| IK | Inverse Kinematics | 速度层广义逆运动学求解 |
| HQP | Hierarchical Quadratic Programming | 不等式约束下的层级优化扩展 |
| WBC | Whole-Body Control | 全身控制总称 |

## 为什么重要

- 每个任务提供当前误差、期望变化和对应雅可比。最高优先级任务先求一个关节速度解；下一任务只在前一任务雅可比的零空间里修正，依次向下堆叠。若低优先级目标与高优先级目标冲突，它只能完成可兼容的部分。关节限位、可视性、避碰等约束可以通过任务激活和不等式处理加入栈中，任务也能随行为阶段插入、删除或改变优先级。
- 在 HMI 六条技术路线中属于 **工程与实机部署**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P003 |
| 年份 | 2009 |
| 分组 | 工程与实机部署 |
| 开源状态 | 社区实现长期存在（stack-of-tasks 生态）；以官方/社区仓为准 |
| 原文 | 见参考来源 |

## 核心原理

很多人第一次接触Stack of Tasks，会把它理解成“按顺序解几个逆运动学”。这篇论文更重要的地方，是把任务、约束、雅可比、数值求解和机器人状态组织成可动态组合的软件实体，让复杂人形行为不必写成一个不可维护的大控制器。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["Stack of Tasks"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

求解器输入是当前广义位置、任务参考和各feature计算出的几何量，输出是广义速度或其积分后的姿态命令。任务对象将误差、雅可比、增益和激活条件封装在一起，solver只处理层级组合；这一软件边界使“看目标”“双手抓取”“保持质心”可以独立测试。它没有读取物体力或执行器状态，因此若接触任务需要力控，还要接操作空间/逆动力学层。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 社区实现长期存在（stack-of-tasks 生态）；以官方/社区仓为准 |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（社区实现长期存在（stack-of-tasks 生态）；以官方/社区仓为准）。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**Stack of Tasks 应作为 HMI「工程与实机部署」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：社区实现长期存在（stack-of-tasks 生态）；以官方/社区仓为准。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 它解决的是广义逆运动学，不是完整逆动力学。输出首先是满足几何和速度任务的广义速度/轨迹，接触力、力矩饱和、浮基动力学和执行器带宽仍需要下游控制器处理。把SoT生成的可行姿态目标直接等同于真实机器人可执行动作，是常见误用。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

Stack of Tasks 的定位是「速度层广义逆运动学的软件化框架」。与相邻页面的关键区别在于求解层次与约束表达能力：

| 维度 | 本工作（Stack of Tasks） | [OSF](./paper-operational-space-formulation.md) | [hqp](../concepts/hqp.md) | [tsid](../concepts/tsid.md) |
|------|--------------------------|-------------------------------------------------|---------------------------|-----------------------------|
| 方法族 | 可动态插拔的层级广义逆运动学任务栈 | 操作空间运动/力动力学 | 层级二次规划（原生含不等式） | 任务空间逆动力学 |
| 求解输出 | 广义速度/积分后的姿态命令 | 关节力矩 | 各层最优解（速度或力矩，视表述） | 加速度/接触力/关节力矩 |
| 约束处理 | 零空间递归（以等式为主，任务激活/不等式为扩展） | 动态一致映射 | 原生支持限位、避碰等不等式约束 | 动力学 + 接触约束 |
| 是否含接触力 | 否，需下游力控层 | 是，含力方向 | 视问题设定 | 是 |
| 关系/取舍 | 把任务/约束/求解器做成可独立测试的软件实体；不读物体力，接触力控需接 OSF/逆动力学层 | 提供其继承的任务空间动力学前驱 | 用不等式扩展 SoT 的硬约束表达 | 在 SoT 之上补齐动力学与力 |

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [hqp](../concepts/hqp.md)
- [tsid](../concepts/tsid.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [paper-operational-space-formulation](./paper-operational-space-formulation.md)

## 参考来源

- [sources/papers/hmi_p003_hmi-stack-of-tasks.md](../../sources/papers/hmi_p003_hmi-stack-of-tasks.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [代码](https://github.com/stack-of-tasks/sot-core)
- [HMI 逐篇解读 P003](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P003.md)
