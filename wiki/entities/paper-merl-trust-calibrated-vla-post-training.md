---
type: entity
tags: [paper, vla, reinforcement-learning, model-based-rl, world-model, post-training, imagination]
status: complete
updated: 2026-10-22
project_id: merl-trust-calibrated-vla
arxiv: "2610.16517"
related:
  - ../methods/vla.md
  - ../methods/model-based-rl.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/merl_arxiv_2610_16517.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "MERL（arXiv:2610.16517）：信任校准的 VLA 后训练框架，把策略优化与在线更新的想象器（世界模型加奖励代理头）耦合，由模拟器可靠度与块级信任分别控制想象数据比例与回放选择。"
---

# MERL：信任校准的 VLA 后训练

**MERL 让想象器随训练一起更新，并用它的可靠度决定多少想象数据可用，避免错误想象把策略拉偏。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| MERL | Model-Evolving Reinforcement Learning | 模型演化强化学习，本文框架名 |

## 为什么重要

摘要指出，无模型精修不受模拟器偏差影响但交互成本高；基于模型的精修则可能利用不可靠的想象数据。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.16517](https://arxiv.org/abs/2610.16517) |
| 作者 | Wenzhan Li, Yiran Qin, Heng Zhou, Huirui Wang, Yulan Guo, Ruimao Zhang（arXiv 作者列表） |
| 机构 | 未核实（摘要页未列） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **想象器：** 世界模型加奖励代理头，在线更新。
- **阶段级控制：** 模拟器可靠度决定想象数据比例与 rollout 长度。
- **块级控制：** chunk 级信任决定回放选择与优化权重。
- **递归想象：** 用残差预测器处理没有真实后续的递归想象。

## 工程实践

- 未发现官方代码入口；源码运行时序图：**不适用**。

## 评测与结果

据 arXiv 摘要（2026-10-22 核对）：

| 设置 | 结果 |
|------|------|
| 扰动版 LIBERO-PRO 任务 | 平均成功率 **79.7%** |
| 学习速度 | 高阈值学习速度比匹配的无模型对照快 10.9% |
| 外部基线 | 三个训练随机种子下，优于同协议外部基线（摘要未列名称与数字） |
| 真机 | 每任务 50 条示范的固定数据适配，2 个真机任务平均成功率 80.0% |

## 与其他工作对比

| 后训练路线 | 数据来源 | 与 MERL 的差异 |
|------------|----------|----------------|
| 无模型 RL 精修（论文匹配对照） | 只用真实 / 仿真交互 | 无模拟器偏差但交互成本高；MERL 学习速度快 10.9% |
| 固定或弱校准的基于模型精修 | 世界模型想象数据 | 可能利用不可靠想象；MERL 用阶段级可靠度与块级信任限制想象数据比例与权重 |
| [WMPO](./paper-sa-2511-09515-wmpo-world-model-based-policy-optimization-for-v.md) | 世界模型中做 VLA 策略优化 | 同属世界模型后训练；MERL 的区别在于想象器在线演化并带信任校准 |
| [World4RL](./paper-sa-2509-19080-world4rl-diffusion-world-models-for-policy-refin.md) | 扩散世界模型精修策略 | 同上，MERL 额外用残差预测器在无真实后续时估计信任 |

前两行对应摘要所述的两类范式（第一行为论文同协议对照）；后两行为路线层面的定性对比，未在同一协议下比较。

## 结论

**一句话总判：后训练中的想象数据应按模拟器可靠度与片段级信任加权，而不是一概使用。**

1. **可靠度应驱动数据比例** — 摘要以此为核心机制。
2. **片段级信任比整体加权更细** — 块级控制用于回放与权重。
3. **想象器需与策略同步演化** — 在线更新是与固定模拟器的主要区别。

## 局限与风险

- 成功率数字仅取自 arXiv 摘要；外部基线的名称与数值摘要未列出，需回原文核对。
- 残差预测器与奖励代理头的误差如何传播，需看论文方法节（推测）。

## 关联页面

- [VLA 方法总览](../methods/vla.md)
- [基于模型的强化学习](../methods/model-based-rl.md)
- [强化学习](../methods/reinforcement-learning.md)

## 参考来源

- [论文归档](../../sources/papers/merl_arxiv_2610_16517.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.16517](https://arxiv.org/abs/2610.16517)
