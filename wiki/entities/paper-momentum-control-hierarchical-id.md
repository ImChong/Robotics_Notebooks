---
type: entity
tags: ["paper", "wbc", "momentum-control", "inverse-dynamics", "humanoid", "hmi-papers"]
status: complete
updated: 2026-07-31
venue: "HMI curated · 2016"
summary: "Momentum Control（HMI P004）：以质心线/角动量为高层平衡目标，用层级逆动力学在浮基动力学与接触约束内求加速度、接触力与力矩。"
related:
  - ../concepts/centroidal-dynamics.md
  - ../concepts/whole-body-control.md
  - ../concepts/tsid.md
  - ./paper-hmi-stack-of-tasks.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p004_momentum-control-hierarchical-id.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Momentum Control（HMI P004）

**Momentum Control**（*Momentum Control with Hierarchical Inverse Dynamics on a Torque-Controlled Humanoid*，2016）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P004**，主分类为 **工程与实机部署**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

以质心线/角动量为高层平衡目标，用层级逆动力学在浮基动力学与接触约束内求加速度、接触力与力矩。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ID | Inverse Dynamics | 由期望加速度/接触求力矩 |
| CoM | Center of Mass | 线动量相关质心量 |
| WBC | Whole-Body Control | 全身多任务控制 |
| QP | Quadratic Program | 接触与任务约束下的优化求解 |

## 为什么重要

- 控制器根据估计状态、接触集合和任务参考建立一组约束。浮基刚体动力学保证求得的加速度、接触力与关节力矩彼此一致；接触加速度约束让支撑点不随意滑动；摩擦和单边接触限制接触力；力矩与关节约束限制硬件可执行范围。在这些硬约束内，第一层跟踪期望质心/角动量变化，后续层再处理摆动脚、躯干、手和姿态。
- 在 HMI 六条技术路线中属于 **工程与实机部署**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P004 |
| 年份 | 2016 |
| 分组 | 工程与实机部署 |
| 开源状态 | 未作为本库主复现入口核验；以原文为准 |
| 原文 | 见参考来源 |

## 核心原理

力矩控制人形的难点不是算出一个好看的关节轨迹，而是同时满足浮基动力学、接触约束和任务优先级。本文把质心线动量与全身角动量作为高层平衡目标，再用层级逆动力学求关节加速度、接触力和力矩，让“机器人整体怎样运动”先于局部关节姿态。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["Momentum Control"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

决策量不是一条抽象“全身动作”，而是广义加速度、接触力与关节力矩的动力学一致组合。输入来自状态估计器、接触计划和各任务参考，输出最终力矩给真实执行器。动量目标通常由质心/姿态误差和期望外力形成，摆动脚与手等局部任务只能在高优先级动力学和接触可行域内优化。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 未作为本库主复现入口核验；以原文为准 |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（未作为本库主复现入口核验；以原文为准）。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**Momentum Control 应作为 HMI「工程与实机部署」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：未作为本库主复现入口核验；以原文为准。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 论文使用层级逆动力学，而不是把所有任务塞进一个加权目标。这样平衡任务不会因为手部或姿态权重过大而被牺牲。代价是求解器必须可靠处理多层可行域，接触计划错了或上层任务本身不可行时，下层再聪明也救不回来。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [centroidal-dynamics](../concepts/centroidal-dynamics.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [tsid](../concepts/tsid.md)
- [paper-hmi-stack-of-tasks](./paper-hmi-stack-of-tasks.md)

## 参考来源

- [sources/papers/hmi_p004_momentum-control-hierarchical-id.md](../../sources/papers/hmi_p004_momentum-control-hierarchical-id.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [HMI 逐篇解读 P004](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P004.md)
