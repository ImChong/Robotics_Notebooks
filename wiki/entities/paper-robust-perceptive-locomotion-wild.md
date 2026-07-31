---
type: entity
tags: ["paper", "quadruped", "locomotion", "perception", "privileged-learning", "eth", "hmi-papers"]
status: complete
updated: 2026-07-31
arxiv: "2201.08117"
venue: "HMI curated · 2022"
summary: "Robust Perceptive Locomotion（HMI P012）：用循环 Belief Encoder 融合带噪高程图与本体历史，使四足在外感知失效时仍能退回身体反馈、在野外稳健行走。"
related:
  - ../concepts/privileged-training.md
  - ../concepts/terrain-latent-representation.md
  - ./extreme-parkour.md
  - ./anymal.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/hmi_p012_robust-perceptive-locomotion-wild.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Robust Perceptive Locomotion（HMI P012）

**Robust Perceptive Locomotion**（*Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild*，2022，[arXiv:2201.08117](https://arxiv.org/abs/2201.08117)）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P012**，主分类为 **Locomotion与运动先验**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

用循环 Belief Encoder 融合带噪高程图与本体历史，使四足在外感知失效时仍能退回身体反馈、在野外稳健行走。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PPO | Proximal Policy Optimization | 教师策略训练算法 |
| BC | Behavior Cloning | 学生模仿教师动作 |
| DR | Domain Randomization | 破坏高度图以模拟传感失效 |
| Sim2Real | Simulation to Real | 野外部署迁移 |

## 为什么重要

- 第一阶段的教师在仿真中能看到无噪地形、接触和环境参数，用PPO得到高性能策略。第二阶段学生接收真机可得的本体感觉和被系统性破坏的高度采样：随机偏移模拟里程计漂移，大噪声和遮挡模拟传感器失效，局部错误模拟地图异常。循环编码器从历史构造belief，策略模仿教师动作；同时解码器要求belief重建无噪高度与特权状态。
- 在 HMI 六条技术路线中属于 **Locomotion与运动先验**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P012 |
| 年份 | 2022 |
| 分组 | Locomotion与运动先验 |
| 开源状态 | 论文未作为本库主复现入口挂代码；概念影响后续感知 loco 管线 |
| 原文 | https://arxiv.org/abs/2201.08117 |

## 核心原理

常见感知运动管线把点云融合成2.5D高程图，再把地图高度送给控制器。问题是雪、植被、反光、遮挡、位姿漂移和视野外区域都会让地图出现假台阶或空洞。本文没有假设地图总是正确，而是让一个循环Belief Encoder把带噪地形观测与本体感觉历史融合成信念状态，控制器可以在外感知可靠时提前调整，在外感知异常时退回身体反馈。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["Robust Perceptive Locomotion"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

行为克隆损失回答“下一步动作是否像教师”，重建损失回答“中间表示是否保留了与控制有关的环境信息”。编码器中的门控决定多少外感知进入belief，因此策略不必在每一帧都同等相信地图。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 论文未作为本库主复现入口挂代码；概念影响后续感知 loco 管线 |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（论文未作为本库主复现入口挂代码；概念影响后续感知 loco 管线）。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**Robust Perceptive Locomotion 应作为 HMI「Locomotion与运动先验」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：论文未作为本库主复现入口挂代码；概念影响后续感知 loco 管线。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 图的第一层是教师策略。仿真器向教师提供三类输入：速度命令；本体观测，包括机体速度与姿态、关节位置和速度历史、动作历史及四条腿的相位；以及每只脚周围五个半径上的208个无噪高度样本。教师还可以读取足端接触状态、接触力与法向、摩擦系数、腿部碰撞、外力和摆动时间等特权量。高度编码器把四只脚附近的几何压成96维地形latent，特权编码器再生成24维接触与动力学latent，主MLP据此输出16维动作：四条腿的相位修正和12个关节位置残差。PPO奖励让动作跟踪目标速度，同时约束足端净空、碰撞、滑移、力矩和目标平滑。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

| 维度 | 本工作（Robust Perceptive Loco） | [Extreme Parkour](./extreme-parkour.md) | [ANYmal](./anymal.md) | [privileged-training](../concepts/privileged-training.md) | [terrain-latent](../concepts/terrain-latent-representation.md) |
|------|----------------------------------|-----------------------------------------|-----------------------|-----------------------------------------------------------|----------------------------------------------------------------|
| 核心问题 | 外感知（高程图）失效时如何稳健行走 | 极限地形上的敏捷越障 | 四足硬件与运控平台 | 教师-学生训练范式 | 地形几何的潜表示 |
| 外感知处理 | Belief Encoder 融合带噪高程图与本体历史，可退回身体反馈 | 端到端消费机载深度 | 平台本身不限定 | 通用范式，不指定外感知 | 把地形压成 latent 供控制器 |
| 训练范式 | 教师（特权观测）→ 学生 BC + belief 重建 | 教师-学生蒸馏 | 不适用 | 特权信息教师蒸馏到可部署学生 | 表示层概念 |
| 关键假设 | 地图可能错，不假设外感知总正确 | 深度可用且大致可靠 | 硬件假设 | 学生只能拿真机可得观测 | latent 保留控制相关几何 |
| 关系/取舍 | 用重建损失让 belief 保留控制相关信息，换来对传感失效的鲁棒 | 同为感知 loco，但优化敏捷而非鲁棒退化 | 本工作常用验证平台族 | 本工作采用的训练框架 | 本工作 belief 的表示动机 |

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [privileged-training](../concepts/privileged-training.md)
- [terrain-latent-representation](../concepts/terrain-latent-representation.md)
- [extreme-parkour](./extreme-parkour.md)
- [anymal](./anymal.md)

## 参考来源

- [sources/papers/hmi_p012_robust-perceptive-locomotion-wild.md](../../sources/papers/hmi_p012_robust-perceptive-locomotion-wild.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [arXiv:2201.08117](https://arxiv.org/abs/2201.08117)
- [HMI 逐篇解读 P012](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P012.md)
