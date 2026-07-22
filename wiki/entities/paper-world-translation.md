---
type: entity
tags: [paper, sim2real, dynamics-modeling, domain-adaptation, partial-observability, unitree-go2]
status: complete
updated: 2026-07-22
arxiv: "2607.18154"
related: [../concepts/sim2real.md, ../concepts/domain-randomization.md, ../concepts/system-identification.md]
sources: [../../sources/papers/world_translation_arxiv_2607_18154.md]
summary: "World Translation 从已经发生的状态转移反向抽取不可观测动力学，并以无配对域翻译迁移仿真/现实风格，针对突发接触等历史不可辨识的 Sim2Real 误差。"
---

# World Translation：反向动力学提取的 Sim2Real 域翻译

**World Translation** 从观测到的状态转移反向抽取隐含动力学，再以无配对域翻译在仿真与现实间保留动力学内容、迁移域风格。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| Sim2Real | Simulation to Reality | 仿真策略迁移到现实 |
| POMDP | Partially Observable Markov Decision Process | 隐变量造成部分可观测 |
| BDE | Backward Dynamics Extraction | 从已发生转移反推动力学特征 |
| UDT | Unpaired Domain Translation | 无配对仿真/现实特征翻译 |

## 核心洞见

历史编码器默认隐变量在动作发生前已经留下可辨识痕迹；突发接触、摩擦跳变等并不满足该假设。论文改从结果回看原因，把 deterministic-but-imperfect 的模拟器与 accurate-but-underdetermined 的学习动力学互补起来。

```mermaid
flowchart LR
  T[观测状态转移] --> B[反向提取隐动力学]
  B --> C[动力学内容表示]
  C --> U[无配对域翻译]
  U --> S[现实风格的仿真转移]
  S --> P[策略迁移]
```

## 结果与工程价值

- 在人形、四足和机械臂平台上评估，尤其在历史无法恢复隐变量时优于基线。
- Unitree Go2 实机部署显示策略迁移改善。
- 工程上适合作为 domain randomization / system identification 的补充，而不是替代安全约束和在线校准。

## 局限与开源状态

- 摘要未给统一量化幅度，选型前需精读任务定义与消融。
- **源码运行时序图：不适用。** 截至 2026-07-22，arXiv 未列官方项目页或代码。

## 关联页面

- [Sim2Real](../concepts/sim2real.md)
- [Domain Randomization](../concepts/domain-randomization.md)
- [System Identification](../concepts/system-identification.md)

## 推荐继续阅读

- [论文 PDF](https://arxiv.org/pdf/2607.18154)

## 参考来源

- [World Translation 论文归档](../../sources/papers/world_translation_arxiv_2607_18154.md)
