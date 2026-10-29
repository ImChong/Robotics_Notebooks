---
type: entity
project_id: e05-geometry-world-action
tags: [paper, robotics, world-model, manipulation]
status: draft
updated: 2026-10-29
arxiv: "2610.36011"
related:
  - ../concepts/world-action-models.md
  - ../tasks/manipulation.md
  - ./paper-rollback-world-model.md
sources:
  - ../../sources/papers/e05_arxiv_2610_36011.md
summary: "E0.5 用 3D 基础模型联合预测未来几何与动作，探索几何原生世界动作建模。"
---

# E0.5：几何原生世界动作模型

## 一句话定义

E0.5 以三维基础模型为主干，联合预测未来三维状态与机器人动作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World-Action Model | 联合建模世界变化和动作 |
| VLM | Vision-Language Model | 视觉语言模型 |
| 3D | Three-Dimensional | 三维几何表征 |

## 为什么重要

几何直接参与世界状态与动作预测，有别于把三维特征仅作为视频模型或 VLM 的辅助输入。

## 核心机制

3D 基础模型同时承担未来几何与动作预测；在机器人轨迹上联合预训练，使表征学习交互中的几何变化。它与 [ThinkWorld](./paper-thinkworld.md) 的未来视觉特征一致性奖励不同。

## 源码运行时序图

不适用：arXiv 摘要未列项目页或代码地址，官方可运行实现待核实。

## 实验与评测

- RoboTwin 2.0 中未做具身数据预训练时，clean / randomized 成功率分别为 **89.90% / 50.09%**。
- 基于 AgiBotWorld-Beta 超过 16 万条轨迹预训练后，AgileX Piper 和 Zerith H1 真机平均成功率分别提升约 **15 和 7 个百分点**；这是预训练带来的提升，并非绝对成功率。

## 结论

几何预测和动作学习可共享三维基础模型，但结果依赖训练数据与实验设置。

1. 对比时区分 clean 与 randomized 场景。
2. 预训练增益须与相应平台的未预训练模型比较。
3. 实际部署前核实传感器、计算和任务条件。

## 与其他工作对比

[Rollback](./paper-rollback-world-model.md) 以世界模型反事实分支改进策略，E0.5 则关注未来几何和动作的联合预测；两者评测协议不同，不能直接比较数字。

## 工程实践与局限

本页依据论文摘要，实验细节与实现入口需结合全文核查；截至 2026-10-29，arXiv 摘要未列项目地址，开放状态待确认。

## 关联页面

- [世界动作模型](../concepts/world-action-models.md)
- [操作任务](../tasks/manipulation.md)

## 参考来源

- [E0.5 论文摘录](../../sources/papers/e05_arxiv_2610_36011.md)

## 推荐继续阅读

- [E0.5 论文](https://arxiv.org/abs/2610.36011)
