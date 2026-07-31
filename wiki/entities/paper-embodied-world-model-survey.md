---
type: entity
tags: ["paper", "survey", "world-model", "embodied-ai", "hmi-papers"]
status: complete
updated: 2026-07-31
arxiv: "2510.16732"
venue: "HMI curated · 2025"
summary: "Embodied World Model Survey（HMI P072）：综合整理具身世界模型的表示、训练目标、规划/策略用法与评测，帮助区分「会预测画面」与「能支撑决策」。"
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ./paper-dreamer-latent-imagination.md
  - ./worldarena.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p072_embodied-world-model-survey.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Embodied World Model Survey（HMI P072）

**Embodied World Model Survey**（*A Comprehensive Survey on World Models for Embodied AI*，2025，[arXiv:2510.16732](https://arxiv.org/abs/2510.16732)）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P072**，主分类为 **世界模型、VLA与Agent**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

综合整理具身世界模型的表示、训练目标、规划/策略用法与评测，帮助区分「会预测画面」与「能支撑决策」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 世界模型 |
| WAM | World-Action Model | 联合建模世界与动作 |
| MBRL | Model-Based RL | 基于模型的强化学习 |
| VLA | Vision-Language-Action | 常与世界模型级联的上层 |

## 为什么重要

- Decision-coupled模型紧贴某个环境、奖励或策略，目标是为规划、价值估计或策略学习产生足够准确的可操作预测，PlaNet和Dreamer是典型例子。General-purpose模型强调跨场景、跨任务的高保真未来生成，可以是数据引擎、互动模拟器或下游agent的环境。前者可以画面不漂亮但控制有用，后者可以视频逼真但尚不保证动作-物理因果准确。评估时不能拿FVD代替任务成功率，也不能只用单任务回报宣称通用世界模型。
- 在 HMI 六条技术路线中属于 **世界模型、VLA与Agent**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P072 |
| 年份 | 2025 |
| 分组 | 世界模型、VLA与Agent |
| 开源状态 | 综述 |
| 原文 | https://arxiv.org/abs/2510.16732 |

## 核心原理

“世界模型”已同时指向Dreamer式任务潜动力学、自动驾驶占据预测、视频生成模型和通用互动模拟器。这篇综述用三条互相独立的轴重新组织它们：是否与具体决策任务耦合，未来是逐步递推还是全局差分预测，世界状态又用什么空间表示。这比按“扩散/Transformer”列模型更能反映它们能否接入机器人闭环。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["Embodied World Model Survey"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

Sequential simulation/inference从当前状态生成下一状态，再将预测喂回继续展开；它天然适合MPC和任意时域，但每步偏差会累积。Global difference prediction直接从初始状态与时间/动作条件预测远期差分或多个未来，可并行、减少自回归漂移，但对任意长度及中间过程的表达弱。这两类选择与控制时域、计算预算和是否需要中间接触细节直接相关。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 综述 |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（综述）。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**Embodied World Model Survey 应作为 HMI「世界模型、VLA与Agent」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：综述。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- Global latent vector最紧凑，适合快速想象，但容易丢几何和多物体结构；token feature sequence保留更丰富的对象/局部信息，可复用Transformer，计算随token数增长；spatial latent grid把2D/3D空间先验放进表示，更适合占据、遮挡和几何规划；decomposed rendering representation用物体、高斯、神经场等可渲染元素分解场景，提供明确3D结构，但建模与更新成本更高。对loco-manip，只有全局视觉向量往往不足以表示接触点、物体位姿与可通行空间。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

| 维度 | 本工作（EWM 综述） | [Dreamer](./paper-dreamer-latent-imagination.md) | [WorldArena](./worldarena.md) | [world-action-models](../concepts/world-action-models.md) | [generative-world-models](../methods/generative-world-models.md) |
|------|--------------------|--------------------------------------------------|-------------------------------|-----------------------------------------------------------|------------------------------------------------------------------|
| 内容类型 | 综述（表示/目标/用法/评测纵览） | 具体方法（潜想象 MBRL） | 世界模型评测基准/竞技场 | 世界与动作联合建模概念 | 生成式世界模型方法族 |
| 关注点 | 三轴分类：是否耦合决策、递推 vs 全局差分、状态表示 | 潜动力学 + 想象规划 | 统一评测协议 | WAM 建模范式 | 高保真未来生成 |
| 定位 | Decision-coupled 与 general-purpose 两类都覆盖 | 典型 decision-coupled 例子 | 评测视角 | 决策耦合的一支 | 偏 general-purpose 一支 |
| 关键提醒 | 别拿 FVD 代替任务成功率，也别用单任务回报宣称通用 | 画面不必逼真但控制有用 | 提供任务级度量 | 强调动作-世界联合 | 视频逼真不等于物理因果准确 |
| 关系/取舍 | 给出组织框架，把这些方法/基准放到同一坐标系 | 综述所归类的代表方法 | 与综述评测缺口呼应 | 综述整理的一类范式 | 综述整理的一类方法族 |

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [paper-dreamer-latent-imagination](./paper-dreamer-latent-imagination.md)
- [worldarena](./worldarena.md)

## 参考来源

- [sources/papers/hmi_p072_embodied-world-model-survey.md](../../sources/papers/hmi_p072_embodied-world-model-survey.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [arXiv:2510.16732](https://arxiv.org/abs/2510.16732)
- [HMI 逐篇解读 P072](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P072.md)
