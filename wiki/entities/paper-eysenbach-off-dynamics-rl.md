---
type: entity
tags:
  - paper
  - sim2real
  - off-dynamics
  - reinforcement-learning
status: complete
updated: 2026-09-20
arxiv: "2006.13916"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_35_eysenbach-off-dynamics-rl.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "训练分类器判断仿真轨迹在真机上的可信度，无需显式目标动力学模型即可面向迁移训练。"
---

# Off-dynamics reinforcement learning: training for transfer with domain classifiers

**Off-dynamics reinforcement learning: training for transfer with domain classifiers**（[arXiv:2006.13916](https://arxiv.org/abs/2006.13916)）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[35/44]**，归类 **残差学习**。

## 一句话定义

训练分类器判断仿真轨迹在真机上的可信度，无需显式目标动力学模型即可面向迁移训练。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| Sim2Real | Simulation to Real | 仿真到真机 |
| OOD | Out-of-Distribution | 分布外 |

## 为什么重要

- 文内真机微调/离策略迁移代表之一。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **残差学习** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | ICLR 2021 |
| **文内章节** | 残差学习 |
| **要点** | domain classifier 作为辅助信号 shaping 策略学习。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 实验与评测

- **本页为索引级节点**（FreeDof 44 篇梳理 [35/44]）：正文固化文内角色与机制要点，**未转存原文实验表**。
- **回原文须核对的证据**：本页要点是「domain classifier 作为辅助信号 shaping 策略学习」，对应证据是加 / 不加该奖励修正项时目标域回报的对比与消融。
- **读法：** 先对齐平台、任务、指标定义与成功阈值，再读任何数字；勿从公众号摘录外推。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **文内路线** | 归类 **残差学习**；同路线其他节点见 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) |
| **横比口径** | 修正项的收益随 sim/real 动力学差距变化：差距过小时收益被噪声淹没，过大时分类器本身失效。 |
| **开源状态** | **待核实** — 部署 / 复现前以项目页或原文 Code availability 为准 |

## 结论

**当显式动力学模型难建时，判别式迁移信号是可行替代。**

1. 文内角色：残差学习 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：domain classifier 作为辅助信号 shaping 策略学习。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_35_eysenbach-off-dynamics-rl.md](../../sources/papers/freedof_sim2real_35_eysenbach-off-dynamics-rl.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2006.13916)
- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
