---
type: entity
tags: [paper, theory, locomotion, contact, uiuc]
status: complete
updated: 2026-09-14
arxiv: "2609.10273"
related:
  - ../tasks/locomotion.md
  - ../formalizations/contact-complementarity.md
  - ./paper-ebert-nonlinear-normal-modes.md
sources:
  - ../../sources/papers/frame_coded_legged_locomotion_arxiv_2609_10273.md
summary: "Frame-Coded Legged Locomotion（arXiv:2609.10273）：multi-foot contact as finite frame code with erasures; recoverability/noise amplification/stiffness limits; theory only ；截至入库日未见官方代码。"
---

# Frame-Coded Legged Locomotion（arXiv:2609.10273）

**Frame-Coded Legged Locomotion**（*Frame-Coded Legged Locomotion over Noisy Terrain*，[arXiv:2609.10273](https://arxiv.org/abs/2609.10273)）由 **伊利诺伊大学厄巴纳-香槟分校（UIUC）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)）。

## 一句话定义

噪声地形上的帧编码腿式运动理论 — multi-foot contact as finite frame code with erasures。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FEC | Finite Erasure Code | 有限擦除码类比 |
| COM | Center of Mass | 质心动力学 |
| ZMP | Zero Moment Point | 零力矩点稳定 |

## 为什么重要

噪声地形破坏接触模式可辨识性；需信息论视角理解腿式可恢复性。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 伊利诺伊大学厄巴纳-香槟分校（UIUC） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

将多足接触序列编码为 frame code；分析擦除噪声下的可恢复性、噪声放大与刚度约束；给出设计界限无真机策略实验。

### 流程总览

```mermaid
flowchart LR
  contact[接触模式序列] --> code[帧编码]
  noise[地形噪声/擦除] --> code
  code --> bounds[可恢复性界限]
  bounds --> design[刚度与控制设计]
```

## 源码运行时序图

**不适用** — 本文为理论/硬件/系统/数据类工作，arXiv 未提供可运行训练或部署仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 理论结果可指导接触传感器冗余与刚度选型；落地需单独控制器实现。 |

## 实验与评测

理论证明与数值界限；无真实机器人 benchmark。

## 结论

帧编码框架为噪声地形腿式运动提供可恢复性与刚度理论界限。

1. 多足接触可作有限码。
2. 擦除模型刻画感知缺失。
3. 刚度极限联系稳定与信息。
4. 无端到端策略论文。
5. 指导传感器与控制 co-design。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 经验型 perceptive locomotion | 缺可恢复性界限 |
| 纯 ZMP 分析 | 未建模编码噪声 |

## 局限与风险

理论假设与真实摩擦/弹性偏差；无开源代码与真机验证。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [contact-complementarity](../formalizations/contact-complementarity.md)
- [./paper-ebert-nonlinear-normal-modes.md](./paper-ebert-nonlinear-normal-modes.md)

## 参考来源

- [frame_coded_legged_locomotion_arxiv_2609_10273.md](../../sources/papers/frame_coded_legged_locomotion_arxiv_2609_10273.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.10273](https://arxiv.org/abs/2609.10273)
