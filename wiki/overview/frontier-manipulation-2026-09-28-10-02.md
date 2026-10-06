---
type: overview
tags: [frontier-papers, manipulation, vla]
status: complete
updated: 2026-10-06
related:
  - ../tasks/manipulation.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/blogs/frontier_manipulation_2026_09_28_10_02.md
summary: "这篇综述把操作研究从可适应 VLA、动态闭环控制与精确 grounding，延伸到世界模型指导的数据采集、触觉探索、人形移动操作记忆与成功—安全联合评测。"
---

# 【9.28–10.2 前沿论文动态】Manipulation

这篇综述把操作研究从可适应 VLA、动态闭环控制与精确 grounding，延伸到世界模型指导的数据采集、触觉探索、人形移动操作记忆与成功—安全联合评测。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 以视觉、语言条件生成机器人动作的策略 |
| RL | Reinforcement Learning | 通过与环境交互优化机器人行为 |
| WAM | World-Action Model | 联合预测环境变化与机器人动作的模型路线 |

## 为什么重要

- 把一周内的论文动态保留为可追溯来源，同时将每篇研究的细节拆到稳定的独立节点。
- 通过问题、方法与论文原文入口连接，不把综述里不同任务的成功率直接横向排序。
- 未附项目页/仓库的条目只记录论文与摘要；需要复现时再逐一核查官方项目页和开源状态。

## 阅读路线

```mermaid
flowchart LR
  A["适配与感知：VLA / 动态操作 / grounding"] --> B["数据与学习：世界模型 / 触觉探索"] --> C["长程与评测：记忆 / 安全基准"]
```

## 独立论文详情

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [Rho：面向高效适应的 VLA 基础模型](../entities/wiki/entities/paper-rho.md) | [2609.38164](https://arxiv.org/abs/2609.38164) | 通用 VLA 一方面需要从大规模数据获得跨任务能力，另一方面部署到具体机器人后又需 要低成本适应。现有方案通常需要针对目标本体重新进行大量 Fine-Tuning，机器人 |
| [DSDyn-VLA：具有运动感知、未来感知与实时修正的双流动态操作框架](../entities/wiki/entities/paper-dsdyn-vla.md) | [2609.39198](https://arxiv.org/abs/2609.39198) | 现有 VLA 在静态桌面操作上已经很强，但遇到传送带、运动目标等动态场景会同时遇到 三个问题：单帧视觉缺乏运动信息；大型 VLA 推理延迟导致动作输出时目标已经移动；Ac |
| [GroundingPI：基于视觉基元的物理智能 Grounding 基础模型 机 构： XPeng Inc. 、 Peking University 、 The University of Hong Kong 、 UC Berkeley 、 Princeton University、NUS、Tsinghua University、HKUST (GZ) 等](../entities/wiki/entities/paper-groundingpi.md) | [2609.39601](https://arxiv.org/abs/2609.39601) | VLA 和 WAM 大多直接继承通用 VLM 或视频生成模型的视觉 Backbone，但机器人操作 对 Grounding 的要求比普通视觉问答高得多：不仅要知道“杯子在 |
| [RoboCoach：将世界模型作为组合式机器人技能的主动教练](../entities/wiki/entities/paper-robocoach.md) | [2609.39685](https://arxiv.org/abs/2609.39685) | 长程操作策略失败以后，通常继续收集整条任务的 End-to-End Demonstration，但真正的 问题往往只发生在其中某一个子技能。例如“开抽屉→拿物体→放入容器 |
| [触觉好奇心驱动机器人交互](../entities/wiki/entities/paper-tactile-curiosity-drives-robot-interaction.md) | [2609.40134](https://arxiv.org/abs/2609.40134) | 机器人随机探索会浪费大量动作在自由空间；普通不确定性奖励也可能偏好自由空间中的模型未知，而不是能改变抓取、滑动和稳定性的接触经验。 |
| [反事实视频生成实现可扩展人形机器人移动操作](../entities/wiki/entities/paper-prism-real2sim2real.md) | [2609.38172](https://arxiv.org/abs/2609.38172) | 从人类视频学习 Humanoid Loco-Manipulation 很有吸引力，但真正适合训练的数据很难 采：视频既要看清完整人体运动，又要看清人与物体的接触，遮挡不能 |
| [T²Mem：面向机器人的测试时记忆学习](../entities/wiki/entities/paper-t2mem.md) | [2609.36720](https://arxiv.org/abs/2609.36720) | 很多长程 Manipulation 是部分可观测的。例如机器人几分钟前看见某个物体被放进抽屉， 现在当前相机画面已经没有这条信息；仅扩大 observation-hist |
| [SafeVLA-Bench：视觉-语言-动作模型成功率与安全性差距评测基准](../entities/wiki/entities/paper-safevla-bench.md) | [2606.00773](https://arxiv.org/abs/2606.00773) | 当前 VLA Benchmark 主要看任务有没有完成，但“成功”不代表执行过程安全。例如机器 人最后把杯子放到了正确位置，但过程中可能碰倒旁边物体、施加过大的接触力、让 |

## 阅读说明

- 同一篇论文可能出现在两篇文章中；arXiv 编号相同就共用一个详情节点。
- 表格中的问题摘要来自附件综述，原始方法、实验设置与结论请以 arXiv 原文为准。
- 本次附件仅为论文综述，没有为每项工作附独立代码/数据链接；没有根据缺少链接推断其未开源。

## 关联页面

- [【9.28–10.2 前沿论文动态】Manipulation 来源归档](../../sources/blogs/frontier_manipulation_2026_09_28_10_02.md)
- [Manipulation任务页](../tasks/manipulation.md)
- [Loco-manipulation](../tasks/loco-manipulation.md)

## 参考来源

- [【9.28–10.2 前沿论文动态】Manipulation 原文](../../sources/blogs/frontier_manipulation_2026_09_28_10_02.md)
- [微信公众号原文](https://mp.weixin.qq.com/s/QnO4PnLUoCM8olHahf8E6g)

## 推荐继续阅读

- [arXiv](https://arxiv.org/)
