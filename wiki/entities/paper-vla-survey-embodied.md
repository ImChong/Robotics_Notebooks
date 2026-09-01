---
type: entity
tags: ["paper", "survey", "vla", "embodied-ai", "hmi-papers"]
status: complete
updated: 2026-09-01
arxiv: "2405.14093"
venue: "HMI curated · 2024"
summary: "VLA Survey（HMI P071）：系统梳理具身 VLA 的数据、架构、训练与评测维度，便于把「通才策略」主张拆成可比较的技术选择。"
related:
  - ../methods/vla.md
  - ../overview/vla-open-source-repro-landscape-2025.md
  - ./openvla.md
  - ../methods/π0-policy.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p071_vla-survey-embodied.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# VLA Survey（HMI P071）

**VLA Survey**（*A Survey on Vision-Language-Action Models for Embodied AI*，2024，[arXiv:2405.14093](https://arxiv.org/abs/2405.14093)）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P071**，主分类为 **世界模型、VLA与Agent**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

系统梳理具身 VLA 的数据、架构、训练与评测维度，便于把「通才策略」主张拆成可比较的技术选择。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| VLM | Vision-Language Model | 常见初始化主干 |
| IL | Imitation Learning | 主训练范式之一 |
| OXE | Open X-Embodiment | 常用跨本体数据底座 |

## 为什么重要

- 视觉基础表示可以来自CLIP、DINO等预训练，语言编码与多模态对齐给出语义条件；动力学学习、世界模型和RL可以为数据生成、策略预训练或在线规划提供信号；推理和policy steering可以约束动作或生成中间目标。有的方法训练时用世界模型，部署时只保留策略；有的方法每个控制步都用模型规划。因此分类时应标清组件位于训练链还是在线执行链，不能只列模块名字。
- 在 HMI 六条技术路线中属于 **世界模型、VLA与Agent**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P071 |
| 年份 | 2024 |
| 分组 | 世界模型、VLA与Agent |
| 开源状态 | 综述 |
| 原文 | https://arxiv.org/abs/2405.14093 |

## 核心原理

只要系统同时出现视觉、语言和机器人，就容易被统称为VLA。这篇综述的最大作用是把研究分成三条线：支撑VLA的组件，直接预测低层动作的control policy，以及把长指令拆成子任务的task planner。一个系统可以同时包含后两者，但评估指标、数据和实时性要求完全不同。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["VLA Survey"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

低层VLA可用非Transformer、自回归Transformer或扩散/flow matching生成动作，也可接入3D感知、点位动作或独立action expert。真正需要对齐的是：输出是末端、关节还是技能token；是单步还是action chunk；闭环更新频率和端到端延迟是多少；低层控制器承担了哪些跟踪与安全功能；评测时是否换物体、场景、任务和本体。不指明动作接口的“成功率更高”无法说明身体控制层的优势。

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

**VLA Survey 应作为 HMI「世界模型、VLA与Agent」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：综述。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 高层规划可以端到端产生子任务，也可用模块化语言/代码规划器调技能库。语言模型给出语义上正确的步骤，不代表当前机器人能从所在状态完成它。可行的规划器必须读环境状态、技能前置/后置条件和失败反馈，并在技能执行后重新观测。对人形机器人，还需要一层明确的身体接口，把“拿起箱子”变成低层可跟踪、可中止、可恢复的条件，而不是让LLM直接跳到电机。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

| 维度 | 本工作（VLA Survey） | [VLA（方法页）](../methods/vla.md) | [VLA 开源复现全景](../overview/vla-open-source-repro-landscape-2025.md) | [OpenVLA](openvla.md) |
|------|----------------------|------------------------------------|-----------------------------------------------------------------------|-----------------------|
| 类型 | 综述 / 分类框架 | 概念-方法定义页 | 开源复现生态盘点 | 单个具体 VLA 模型 |
| 主要作用 | 拆成组件 / control policy / task planner 三线 | 给出 VLA 范式与动作接口 | 汇总可复现实现、许可与门槛 | 提供开源基线 |
| 关注点 | 数据/架构/训练/评测维度对齐 | 输入输出与动作接口 | 落地复现难度 | 权重 + 微调路径 |
| 关系/取舍 | 引用列表非选型排名；须标清组件在训练链还是执行链 | Survey 抽象的对象之一 | Survey 的工程落地补充 | Survey 收录的实例 |

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [vla](../methods/vla.md)
- [vla-open-source-repro-landscape-2025](../overview/vla-open-source-repro-landscape-2025.md)
- [openvla](./openvla.md)
- [π0-policy](../methods/π0-policy.md)

## 参考来源

- [sources/papers/hmi_p071_vla-survey-embodied.md](../../sources/papers/hmi_p071_vla-survey-embodied.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [arXiv:2405.14093](https://arxiv.org/abs/2405.14093)
- [HMI 逐篇解读 P071](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P071.md)
