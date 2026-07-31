---
type: entity
tags: ["paper", "amp", "quadruped", "locomotion", "imitation", "hmi-papers"]
status: complete
updated: 2026-07-31
arxiv: "2203.15103"
venue: "HMI curated · 2022"
summary: "AMP Locomotion（HMI P023）：在四足上证明：保留速度任务奖励、用短段犬类动作 AMP 即可替代大量手工步态塑形项，并部署到 Unitree A1。"
related:
  - ./paper-amp-survey-01-amp.md
  - ../methods/amp-reward.md
  - ../methods/ase.md
  - ../tasks/locomotion.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/hmi_p023_amp-locomotion-quadruped-rewards.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# AMP Locomotion（HMI P023）

**AMP Locomotion**（*Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions*，2022，[arXiv:2203.15103](https://arxiv.org/abs/2203.15103)）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P023**，主分类为 **Locomotion与运动先验**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

在四足上证明：保留速度任务奖励、用短段犬类动作 AMP 即可替代大量手工步态塑形项，并部署到 Unitree A1。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AMP | Adversarial Motion Prior | 用判别器替代复杂步态塑形奖励 |
| PPO | Proximal Policy Optimization | 策略优化 |
| IK | Inverse Kinematics | 犬类动作重定向到 A1 |
| PD | Proportional–Derivative | 30 Hz 策略下的关节跟踪 |

## 为什么重要

- 作者先把约4.5秒的德国牧羊犬运动重定向到Unitree A1。重定向使用逆运动学求机器人关节角，再以前向运动学检查脚端等关键部位的位置，关节与末端速度由相邻帧差分得到。动作片段覆盖慢速踱步、快步、小跑和转向，它们不带“当前应跟踪多少速度”的逐帧标签，只向判别器提供自然四足运动的短时状态转移。
- 在 HMI 六条技术路线中属于 **Locomotion与运动先验**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P023 |
| 年份 | 2022 |
| 分组 | Locomotion与运动先验 |
| 开源状态 | 方法复用 AMP/ASE 生态；本篇以 A1 真机验证为主 |
| 原文 | https://arxiv.org/abs/2203.15103 |

## 核心原理

这篇工作的重点不是再次介绍AMP，而是做一个很有工程意义的检验：四足步态奖励通常包含抬脚、落脚、躯干姿态、对称性和能耗等大量规则，能否只保留速度任务奖励，再让一小段动作数据负责运动风格？作者用约4.5秒德国牧羊犬动作训练Unitree A1，并把策略部署到真机。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["AMP Locomotion"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

每个控制周期，Actor读取A1关节角、关节速度、机身方向、上一时刻动作，以及用户给出的前向速度、侧向速度和偏航角速度命令。速度命令范围覆盖后退到快速前进、横移和左右转向。Actor使用三层MLP输出十二个关节目标角，策略以30 Hz运行，底层PD控制器把目标角转换为电机力矩。真机执行后的关节和机身状态回到下一周期，形成速度命令到关节动作的闭环。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 方法复用 AMP/ASE 生态；本篇以 A1 真机验证为主 |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（方法复用 AMP/ASE 生态；本篇以 A1 真机验证为主）。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**AMP Locomotion 应作为 HMI「Locomotion与运动先验」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：方法复用 AMP/ASE 生态；本篇以 A1 真机验证为主。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 训练仍有两类奖励。速度跟踪奖励要求机器人完成当前线速度和角速度命令；AMP判别器比较犬类参考转移与策略 rollout 转移，把接近示范节奏和姿态的结果变成风格奖励。策略因此可以在速度连续变化时自行选择和过渡步态，而不需要为踱步、小跑和转向分别编写状态机。这里“替代复杂奖励”指减少手工设计的抬脚高度、步态对称和姿态塑形项，并不意味着可以删除速度跟踪、动作平滑、关节安全与终止条件。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

| 维度 | 本工作（AMP Locomotion） | [AMP 原始工作](./paper-amp-survey-01-amp.md) | [amp-reward](../methods/amp-reward.md) | [Robot Parkour](./paper-robot-parkour-learning.md) |
|------|--------------------------|----------------------------------------------|----------------------------------------|----------------------------------------------------|
| 方法族 | 判别器风格奖励 + 速度任务奖励 | 判别器风格奖励（仿真角色动画） | AMP 判别器作为通用风格奖励项 | 软→硬动力学约束课程 + 多专家 DAgger 蒸馏 |
| 风格/技能来源 | 约 4.5s 犬类动作重定向到 A1 | 人体/角色 mocap 片段 | 任意参考转移数据集 | 无参考动作，纯任务奖励驱动探索 |
| 关键假设 | 保留速度跟踪，仅让短段动作负责步态风格 | 用示范约束风格、任务奖励管目标 | 状态转移分布可由判别器匹配 | 障碍先可穿透以获得梯度，再恢复真实接触 |
| 载体/验证 | Unitree A1 真机，速度命令闭环 | 仿真物理角色为主 | 方法组件，不绑定具体机器人 | A1/Go1 真机，深度视觉越障 |
| 关系/取舍 | 证明 AMP 可替代四足手工步态塑形，但仍需速度、平滑与安全项 | 本工作的方法基础 | 本工作复用的奖励机制 | 同为「减少手工奖励」路线，但走课程而非示范 |

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [paper-amp-survey-01-amp](./paper-amp-survey-01-amp.md)
- [amp-reward](../methods/amp-reward.md)
- [ase](../methods/ase.md)
- [locomotion](../tasks/locomotion.md)

## 参考来源

- [sources/papers/hmi_p023_amp-locomotion-quadruped-rewards.md](../../sources/papers/hmi_p023_amp-locomotion-quadruped-rewards.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [arXiv:2203.15103](https://arxiv.org/abs/2203.15103)
- [HMI 逐篇解读 P023](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P023.md)
