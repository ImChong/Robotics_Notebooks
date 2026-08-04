---
type: entity
tags: ["paper", "loco-manipulation", "rl", "quadruped", "hmi-papers"]
status: complete
updated: 2026-07-31
arxiv: "2210.10044"
venue: "HMI curated · 2022"
summary: "Deep Whole-Body Control（HMI P042）：用 Advantage Mixing 平衡移动与操作梯度，并配合在线适应估计环境变化，使统一策略在共享身体上协调两类行为。"
related:
  - ../tasks/loco-manipulation.md
  - ./paper-visual-whole-body-control-vbc.md
  - ../concepts/whole-body-control.md
  - ../methods/reinforcement-learning.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p042_deep-whole-body-control-loco-manip.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Deep Whole-Body Control（HMI P042）

**Deep Whole-Body Control**（*Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion*，2022，[arXiv:2210.10044](https://arxiv.org/abs/2210.10044)）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P042**，主分类为 **LocoManip**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

用 Advantage Mixing 平衡移动与操作梯度，并配合在线适应估计环境变化，使统一策略在共享身体上协调两类行为。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 移动与操作统一策略 |
| RL | Reinforcement Learning | 端到端策略学习 |
| PPO | Proximal Policy Optimization | 常见优化器 |
| Sim2Real | Simulation to Real | 真机迁移 |

## 为什么重要

- 策略输入包含基座和关节状态、足端接触、上一动作、末端位置姿态命令、机身速度命令以及环境latent，输出腿和臂的目标关节位置。作者分别计算manipulation advantage与locomotion advantage：训练初期让臂动作更多受操作优势更新、腿动作更多受移动优势更新，使两个子任务先形成有效探索；随后逐渐混合总优势，让全身学会协调。没有这一过程时，策略很容易停在原地追末端，因为走路初期会暂时降低操作回报。
- 在 HMI 六条技术路线中属于 **LocoManip**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P042 |
| 年份 | 2022 |
| 分组 | LocoManip |
| 开源状态 | 部分/社区复现线索需按原文与项目页再核 |
| 原文 | https://arxiv.org/abs/2210.10044 |

## 核心原理

四足加机械臂常用两个独立控制器：腿追速度，臂追末端，两者通过基座扰动被动耦合。本文用一个策略同时读取腿、臂、基座、末端目标和移动命令，直接输出18个关节目标。真正的难点不是网络维度，而是训练早期操作回报和移动回报会把策略拉向不同局部最优。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["Deep Whole-Body Control"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

统一策略的收益不是形式上的“一个网络”，而是腿可以移动和倾斜基座扩大机械臂工作空间，手臂受力时腿也能主动补偿。论文通过独立策略、未协调单策略和完整方法对比，说明协调来自共享状态与联合优化，而不是简单拼接输出。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 部分/社区复现线索需按原文与项目页再核 |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（部分/社区复现线索需按原文与项目页再核）。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**Deep Whole-Body Control 应作为 HMI「LocoManip」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：部分/社区复现线索需按原文与项目页再核。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 末端目标描述的是手应该到达的位姿，并不等于系统显式知道杯子或按钮的完整动力学。接触效果主要通过末端误差、基座响应和任务过程间接体现；抓手开合仍由外部逻辑给出。因而擦拭、按压等成功条件需要由任务层定义，低层策略负责在移动和受力时保持末端可达与身体稳定。遇到物体滑移或目标检测变化，论文中的底座不会自己重新识别物体并规划新抓取点。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

| 维度 | 本工作（Deep WBC） | [VBC](./paper-visual-whole-body-control-vbc.md) | [强化学习（方法）](../methods/reinforcement-learning.md) |
|------|--------------------|--------------------------------------------------|----------------------------------------------------------|
| 方法族 | 单一统一策略端到端输出 18 个腿臂关节目标 | 视觉高层 + 全身低层的两频分层 | 端到端策略学习的通用范式 |
| 感知输入 | 本体状态、接触、末端命令与环境 latent，无显式视觉 | 高层依赖物体 mask/分割深度做视觉闭环 | 不限定观测形式 |
| 协调机制 | Advantage Mixing 分离再混合移动/操作优势 + 在线适应 | 靠高低层分工与蒸馏解耦难度 | 以奖励与优势估计驱动行为 |
| 关键假设 | 早期需分别引导腿/臂探索，避免停在原地追末端 | 假设分层可降低训练与 Sim2Real 难度 | 假设奖励可引导目标行为 |
| 关系/取舍 | 结构更简、协调更强，但无视觉、难感知物体几何 | 引入视觉换取物体级闭环，但依赖分割稳定 | 为本工作的优势混合训练提供理论背景 |

任务背景见 [移动操作](../tasks/loco-manipulation.md)。

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [paper-visual-whole-body-control-vbc](./paper-visual-whole-body-control-vbc.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [sources/papers/hmi_p042_deep-whole-body-control-loco-manip.md](../../sources/papers/hmi_p042_deep-whole-body-control-loco-manip.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [arXiv:2210.10044](https://arxiv.org/abs/2210.10044)
- [HMI 逐篇解读 P042](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P042.md)
