---
type: entity
tags:
  - paper
  - survey
  - space-robotics
  - isru
  - autonomous-systems
  - casia
status: complete
updated: 2026-08-26
arxiv: "2608.21358"
code: https://github.com/OpenSpace-Lab/Space-Mining-with-Robotics-List
related:
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
  - ../methods/reinforcement-learning.md
  - ../overview/open-source-8-papers-technology-map.md
  - ./paper-reward-free-continual-adaptation-space.md
sources:
  - ../../sources/papers/space_mining_with_robotics_arxiv_2608_21358.md
  - ../../sources/sites/space-mining-openspace-lab.md
  - ../../sources/repos/space-mining-with-robotics-list.md
  - ../../sources/blogs/wechat_embodied_station_8_papers_open_source_2026-08-25.md
summary: "太空采矿机器人综述（arXiv:2608.21358，CASIA/OpenSpace Lab）：六阶段勘探–采样–提取架构 + 开放研究清单；强调跨阶段自主与验证基础设施。"
---

# Space Mining with Robotics：太空采矿机器人综述

**Mining beyond Earth with Space Robots: Exploration, Sampling, and Extraction**（[arXiv:2608.21358](https://arxiv.org/abs/2608.21358)，[开放清单](https://github.com/OpenSpace-Lab/Space-Mining-with-Robotics-List)）由 **中国科学院自动化研究所（CASIA）/ OpenSpace Lab** 等联合撰写：系统梳理太空资源利用（ISRU）背景、政策与商业生态，提出 **六阶段自主采矿架构**，并维护持续更新的研究资源库。

## 一句话定义

**太空采矿的真正门槛不是单台机器人，而是跨阶段自主系统与可验证的数据、仿真、测试基础设施。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ISRU | In-Situ Resource Utilization | 原位资源利用与推进剂/建造材料就地获取 |
| NEA | Near-Earth Asteroid | 近地小行星采矿目标 |
| LRO | Lunar Reconnaissance Orbiter | 月面遥感等高成熟度测绘参考 |
| ³He | Helium-3 | 月球高价值聚变燃料候选资源 |
| CAST | China Association for Science and Technology | 2025 将太空采矿列为十大产业技术挑战之一 |

## 为什么重要

- **战略与商业交汇：** 月南极水冰、³He 与小行星金属牵动多国政策与商业实体布局。
- **自主性刚需：** 通信时延（月面 ~1.3 s、火星分钟级）使遥操作难以支撑大规模开采闭环。
- **工程可迁移：** 地球露天采矿自主化提供部分范式，但微重力、粉尘、辐射与稀缺试验环境带来根本差异。
- **开放清单价值：** 论文配套 GitHub 清单聚合政策、文献、任务数据与仿真入口，降低领域入门与 lint 跟进成本。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 中国科学院自动化研究所（CASIA）/ OpenSpace Lab 等 |
| **类型** | 综述 + 开放资源策展 |
| **开源** | **已开源** — `OpenSpace-Lab/Space-Mining-with-Robotics-List`（研究导航仓） |

## 六阶段架构

| 阶段 | 内容 |
|------|------|
| 1 | 遥感选址（remote sensing for target identification） |
| 2 | 原位精细探测（in situ robotic detection） |
| 3 | 单机器人小规模采样 |
| 4 | 多机器人规模化挖掘 |
| 5 | 自主资源提取 |
| 6 | 原位建造或地面运输 |

### 流程总览

```mermaid
flowchart LR
  RS[遥感选址] --> IS[原位探测]
  IS --> S1[单机器人采样]
  S1 --> M[多机器人挖掘]
  M --> EX[自主提取]
  EX --> USE[原位建造/运输]
```

## 工程实践

| 项 | 建议 |
|----|------|
| **入门** | 从开放清单按「任务数据 → 地球类比 → 仿真」顺序建立阅读路径 |
| **算法验证** | 优先用地外任务遥测 + 地球类比数据集做感知/规划回归，再上高保真仿真 |
| **系统拆分** | 按六阶段划分子系统边界，避免把「采样 demo」误当作「提取闭环」 |
| **政策风险** | 跟踪 Artemis Accords 与各国国内立法对资源权属的定义差异 |

## 局限与风险

- 综述不替代单篇感知/操作论文的复现细节。
- 开放清单依赖社区维护，链接失效需定期 lint。
- 大规模提取阶段的能源、热控与粉尘管理仍是开放难题。

## 评测

本工作为 **领域综述 + 资源策展**，不以单一 benchmark 成功率为主指标；价值在于架构清晰度与资源覆盖度。

## 结论

**太空采矿研究应被读成跨阶段自主系统问题，而非孤立机器人技能集合。**

- 六阶段架构把勘探、采样、提取串成可验证价值链
- 通信时延与高发射成本倒逼高自主与 ISRU
- 地球类比数据与高保真仿真是算法落地前置条件
- 开放清单是持续跟进政策、任务与 benchmark 的实用入口
- 单点机器人 demo 不能代表提取–建造闭环成熟度

## 源码运行时序图

| 项 | 说明 |
|----|------|
| **源码运行时序图** | **不适用**（开放清单为策展仓，非可运行训练/仿真管线） |

## 与其他页面的关系

- [Sim2Real](../concepts/sim2real.md) — 地外验证与域随机化语境
- [manipulation](../tasks/manipulation.md) — 采样与抓取子能力
- [reinforcement-learning](../methods/reinforcement-learning.md) — 高自主决策层
- [open-source-8-papers-technology-map](../overview/open-source-8-papers-technology-map.md) — 公众号索引
- [无奖励持续适应](./paper-reward-free-continual-adaptation-space.md) — 地外故障后无奖励在线适应（DreamerV3 × SRB）

## 参考来源

- [space_mining_with_robotics_arxiv_2608_21358](../../sources/papers/space_mining_with_robotics_arxiv_2608_21358.md)
- [space-mining-openspace-lab](../../sources/sites/space-mining-openspace-lab.md)
- [space-mining-with-robotics-list](../../sources/repos/space-mining-with-robotics-list.md)
- [wechat_embodied_station_8_papers_open_source_2026-08-25](../../sources/blogs/wechat_embodied_station_8_papers_open_source_2026-08-25.md)

## 推荐继续阅读

- [arXiv:2608.21358](https://arxiv.org/abs/2608.21358)
- [Space Mining with Robotics List](https://github.com/OpenSpace-Lab/Space-Mining-with-Robotics-List)
