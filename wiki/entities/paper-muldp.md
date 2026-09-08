---
type: entity
tags: ['paper', 'quadruped', 'fudan', 'diffusion-policy', 'parkour', 'navigation']
status: complete
updated: 2026-09-07
arxiv: "2609.03984"
summary: "MulDP（arXiv:2609.03984，复旦）：视觉+本体+目标扩散生成速度指令；QPND 数据集；仿真 SR 89.7%；未见官方代码。"
related:
  - ../tasks/locomotion.md
  - ../methods/diffusion-policy.md
  - ./paper-contact-guided-exploration-locomanipulation.md
sources:
  - ../../sources/papers/muldp_arxiv_2609_03984.md
---

# MulDP：四足跑酷自主导航扩散策略

**MulDP**（[arXiv:2609.03984](https://arxiv.org/abs/2609.03984)）由 **复旦大学智能机器人与先进制造学院** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)）。

## 一句话定义

四足跑酷导航要 **提前加速、细调速度**——MulDP 用扩散直接出 **时序连贯的速度命令**，而不是只避障 waypoint。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MulDP | Multimodal Diffusion Policy | 本文多模态扩散导航策略 |
| QPND | Quadruped Parkour Navigation Dataset | 宣称首个跑酷导航多模态集 |
| DME | Decision Memory Encoder | 近 5 步规划记忆编码器 |

## 为什么重要

模块化建图规划难做 **动态跑酷**；纯视觉 waypoint 与 **机身几何/动力学** 脱节；E2E RL/VLA 成本高。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 复旦大学智能机器人与先进制造学院 |
| **开源** | 见 [工程实践](#工程实践) |

## 核心原理

三编码器：历史深度+本体 Transformer、当前深度 CNN、决策记忆 MLP；条件扩散 denoise **未来速度 horizon**；5 Hz 重规划，首命令送低层 locomotion policy。QPND 在 Isaac Sim 采集+增广。

### 流程总览

```mermaid
flowchart LR
  depth[历史/当前深度] --> enc[多模态编码]
  prop[本体] --> enc
  goal[目标] --> enc
  enc --> diff[扩散速度序列]
  diff --> loco[低层运动策略]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-07** 无可运行官方代码（或本文为硬件/协议类工作）。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 见论文摘录与项目页核查结论 |
| 复现入口 | 以 arXiv 为准 |

## 实验与评测

| 指标 | MulDP | NavDP* |
|------|-------|--------|
| SR | **89.7%** | 59.0% |
| w/o 本体 | 69.5% | — |
| w/o DME | 7.4% SR | — |

## 结论

跑酷导航需要 **本体+决策记忆+扩散时序**；QPND 与 MulDP 是四足 **自主穿越** 方向的实用组合。

1. 可与全局规划器叠用。
2. 消融：去 DME 几乎不收敛到目标。
3. 真机实验论文宣称有效（见原文）。
4. **QPND 未见公开下载**。
5. **代码未开源**。

## 与其他工作对比

跑酷导航的分歧在 **上层给低层下发什么**——路点、还是带时序的速度：

| 路线 | 上层输出 | 是否读本体 | 与机身动力学的耦合 | 与本文 |
|------|----------|------------|--------------------|--------|
| **MulDP** | **未来速度 horizon**（扩散生成，5 Hz 重规划，取首命令） | **是**（本体 + 决策记忆 DME） | 强——能「提前加速」 | 本页；消融显示去 DME 后 SR 塌到 7.4% |
| [NavDP](./paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md) | 扩散导航轨迹 | 弱 | 弱 | 本文自报对照：SR **59.0% → 89.7%**，但那是 **作者复现的 NavDP\***，非原作者同台结果 |
| 模块化建图 + 规划 | 路点 / 路径 | 否 | 弱——规划器不知道机身能否起跳 | 页首动机：难做动态跑酷 |
| 端到端 RL / VLA | 直接出动作或速度 | 是 | 强 | 页首动机：训练与数据成本高；MulDP 想以扩散 + 仿真数据集换掉这份成本 |
| [Contact-Guided Exploration](./paper-contact-guided-exploration-locomanipulation.md) | 全身 loco-manip 动作 | 是 | 强 | 同为四足上层，但解的是 **接触稀疏**，MulDP 解的是 **导航时序** |

**证据强度提醒：** 89.7% 是 **仿真** 数字，QPND 数据集与代码 **均未公开**，NavDP 对照为本文自行复现——把这张表读成「方向对照」而非「排行榜」。可迁移的判断是：**四足导航上层若不读本体、不出时序速度，跑酷类任务会系统性吃亏**。

## 局限与风险

依赖仿真数据与特定低层 policy；泛化地形未完全展开。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [paper-contact-guided-exploration-locomanipulation.md](./paper-contact-guided-exploration-locomanipulation.md)
- [NavDP](./paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md) — 本文主要对照的扩散导航基线

## 参考来源

- [muldp_arxiv_2609_03984.md](../../sources/papers/muldp_arxiv_2609_03984.md)
- [公众号周更 21 篇索引](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.03984](https://arxiv.org/abs/2609.03984)
