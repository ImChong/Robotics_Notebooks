---
type: entity
tags: ["paper", "humanoid", "atlas", "mpc", "state-estimation", "boston-dynamics", "hmi-papers"]
status: complete
updated: 2026-07-31
venue: "HMI curated · 2016"
summary: "Atlas Locomotion（HMI P005）：把足步/全身规划、状态估计与高频反馈控制放进同一条 Atlas 闭环，强调坐标系、频率与状态接口必须统一。"
related:
  - ./boston-dynamics.md
  - ../concepts/whole-body-control.md
  - ../tasks/humanoid-locomotion.md
  - ../concepts/state-estimation.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p005_atlas-locomotion-optimization-stack.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Atlas Locomotion（HMI P005）

**Atlas Locomotion**（*Optimization-based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot*，2016）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P005**，主分类为 **工程与实机部署**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

把足步/全身规划、状态估计与高频反馈控制放进同一条 Atlas 闭环，强调坐标系、频率与状态接口必须统一。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LQR | Linear Quadratic Regulator | 沿名义轨迹的反馈修正 |
| MPC | Model Predictive Control | 相关的在线重规划思路族 |
| WBC | Whole-Body Control | 全身跟踪与接触执行 |
| CoM | Center of Mass | 行走规划常用平衡量 |

## 为什么重要

- 系统可以由足步规划器先决定接触位置和时序，再生成行走参考；也可以由全身运动规划器处理更复杂的身体与环境约束。规划结果不是直接发给电机，而是形成名义状态和输入轨迹。控制层围绕这条轨迹使用时变LQR等反馈，根据实时状态偏差修正动作；必要时LQR解可在线重新计算，降低真实机器人偏离名义轨迹后的敏感性。
- 在 HMI 六条技术路线中属于 **工程与实机部署**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P005 |
| 年份 | 2016 |
| 分组 | 工程与实机部署 |
| 开源状态 | 未开源（工业/实验室闭源系统论文） |
| 原文 | https://doi.org/10.1007/s10514-015-9479-3 |

## 核心原理

动态人形系统常见的问题是每个模块单独看都合理：足步规划能给路线，轨迹优化能给全身动作，状态估计能给姿态，控制器能跟踪。但只要坐标系、频率、状态定义或延迟没有统一，整机仍然走不起来。本文的价值正是把规划、估计和控制放进同一条Atlas闭环，而不是只报告一个局部算法。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["Atlas Locomotion"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

状态估计与控制在高频闭环运行，融合机器人传感信息并给控制器提供浮基状态。规划线程和控制线程时间尺度不同：前者可以较慢地解决未来接触和轨迹，后者必须持续吸收估计误差与扰动。系统设计的关键是明确每一层消费什么状态、输出什么参考，以及旧计划在新状态下是否仍然有效。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 未开源（工业/实验室闭源系统论文） |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（未开源（工业/实验室闭源系统论文））。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**Atlas Locomotion 应作为 HMI「工程与实机部署」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：未开源（工业/实验室闭源系统论文）。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 计划的接口应包含接触位置/时序、名义状态和控制轨迹以及有效时间窗；估计器输出基座、速度和接触状态；反馈控制器再给关节执行命令。新估计与名义轨迹偏差超过阈值时，系统需要从当前状态重新规划，而不是继续沿旧轨迹加大反馈。不同线程应使用统一时钟并标记计划版本，防止控制器拿到一半新、一半旧的数据。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [boston-dynamics](./boston-dynamics.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [state-estimation](../concepts/state-estimation.md)

## 参考来源

- [sources/papers/hmi_p005_atlas-locomotion-optimization-stack.md](../../sources/papers/hmi_p005_atlas-locomotion-optimization-stack.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [DOI](https://doi.org/10.1007/s10514-015-9479-3)
- [HMI 逐篇解读 P005](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P005.md)
