---
type: entity
tags: ["paper", "survey", "loco-manipulation", "humanoid", "hmi-papers"]
status: complete
updated: 2026-07-31
arxiv: "2501.02116"
venue: "HMI curated · 2025"
summary: "Humanoid Loco-Manipulation Survey（HMI P069）：按控制层、任务类型与真实证据整理人形移动操作研究，统一术语并指出触觉、复杂接触与系统评测缺口。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../tasks/loco-manipulation.md
  - ../concepts/whole-body-control.md
  - ../../roadmap/motion-control.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p069_humanoid-loco-manipulation-survey.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Humanoid Loco-Manipulation Survey（HMI P069）

**Humanoid Loco-Manipulation Survey**（*Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning*，2025，[arXiv:2501.02116](https://arxiv.org/abs/2501.02116)）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P069**，主分类为 **LocoManip**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

按控制层、任务类型与真实证据整理人形移动操作研究，统一术语并指出触觉、复杂接触与系统评测缺口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LocoManip | Loco-Manipulation | 移动与操作耦合任务 |
| WBC | Whole-Body Control | 全身控制层 |
| RL | Reinforcement Learning | 学习类方法分支 |
| MPC | Model Predictive Control | 模型控制分支 |

## 为什么重要

- 接触规划决定什么时候用哪只脚、哪只手或身体哪个部位接触什么；运动规划/最优控制在这些模式下求身体、物体和力的轨迹；MPC在短视界内持续重规划；WBC用较高频率将轨迹变成满足动力学、接触锥和关节限制的力矩/位置命令。高保真模型带来约束可解释性，也带来接触组合爆炸和在线计算压力；这就是为什么实际系统常用形心MPC + 局部全身控制的预测-反应层级。
- 在 HMI 六条技术路线中属于 **LocoManip**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P069 |
| 年份 | 2025 |
| 分组 | LocoManip |
| 开源状态 | 综述（开源条目随引用工作变化） |
| 原文 | https://arxiv.org/abs/2501.02116 |

## 核心原理

这篇综述的价值不在于给出一个新算法，而是把长期分散在接触规划、运动规划、MPC/WBC、强化学习、模仿学习、基础模型和触觉感知中的工作放到一条系统链上。它还做了一个重要澄清：移动操作关心机器人边移动边操作，whole-body manipulation关心如何把手、胸、腿等所有可用表面变成接触，whole-body loco-manipulation则同时要求两者。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["Humanoid Loco-Manipulation Survey"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

RL能在仿真里通过大量试错学到扰动恢复和难手工设计的动态行为，但高维、稀疏奖励的loco-manip纯探索成本高；模仿学习用人类或规划器示范缩小搜索空间，但又受重定向、物理可行性和数据覆盖限制。综述明确指出，sim-to-real RL仍依赖仿真动力学模型，它与模型控制不矛盾。更有希望的组合是：规划/约束给结构和安全边界，学习策略处理模型误差、鲁棒性和高维经验。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 综述（开源条目随引用工作变化） |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（综述（开源条目随引用工作变化））。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**Humanoid Loco-Manipulation Survey 应作为 HMI「LocoManip」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：综述（开源条目随引用工作变化）。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 全身loco-manip中，许多关键状态不能从视觉和关节编码器可靠推断：手掌是否即将滑移、胸部接触承受多大法向/切向力、脚底压力中心是否已越界。全身触觉可用于接触检测、力估计、模式切换和反应控制，但目前传感器耐久性、校准、带宽、覆盖面积与统一控制架构都不成熟。这一缺口说明仅扩大视觉-语言模型并不会自动解决接触交互。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

| 维度 | 本工作（LocoManip 综述） | [161 篇技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md) | [whole-body-control](../concepts/whole-body-control.md) | [loco-manipulation](../tasks/loco-manipulation.md) |
|------|--------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------|----------------------------------------------------|
| 内容类型 | 学术综述（控制/规划/学习纵览） | 本库对同领域的技术地图导览 | 单一控制层概念 | 任务定义节点 |
| 覆盖范围 | 接触规划、运控、MPC/WBC、RL、IL、基础模型、触觉 | 领域论文全景与聚类 | 仅全身控制这一层 | 移动+操作耦合任务本身 |
| 组织维度 | 按控制层 × 任务类型 × 真实证据，统一术语 | 按主题/技术聚类 | 按控制原理 | 按任务接口 |
| 主要产出 | 术语澄清 + 触觉/复杂接触/系统评测缺口 | 检索与定位入口 | 原理解释 | 任务边界界定 |
| 关系/取舍 | 提供领域框架，不给新算法；澄清 whole-body manip 与 loco-manip 区分 | 与本综述互为「地图 vs 导览」 | 综述所整理的 WBC 层的下钻页 | 综述所服务的任务节点 |

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [humanoid-loco-manip-161-papers-technology-map](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [motion-control](../../roadmap/motion-control.md)

## 参考来源

- [sources/papers/hmi_p069_humanoid-loco-manipulation-survey.md](../../sources/papers/hmi_p069_humanoid-loco-manipulation-survey.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [arXiv:2501.02116](https://arxiv.org/abs/2501.02116)
- [HMI 逐篇解读 P069](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P069.md)
