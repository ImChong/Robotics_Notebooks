---
type: entity
tags: ['paper', 'quadruped', 'embodied-intelligence', 'tum', 'dlr', 'kaist', 'hardware']
status: complete
updated: 2026-09-07
arxiv: "2609.00539"
summary: "TUM/DLR/KAIST（arXiv:2609.00539）：高柔顺 SEA 四足 eBert 识别 6 个 NNM；黑箱步长优化使各模态发展为不同速度步态；具身智能 proof-of-concept。"
related:
  - ../tasks/locomotion.md
  - ../concepts/embodied-foundation-model-hardware-codesign.md
  - ./paper-qlaun.md
sources:
  - ../../sources/papers/ebert_nonlinear_normal_modes_arxiv_2609_00539.md
---

# eBert：非线性正规模涌现四足步态

**eBert NNM Gaits**（[arXiv:2609.00539](https://arxiv.org/abs/2609.00539)）由 **慕尼黑工业大学（TUM）、德国航空航天中心（DLR）、韩国科学技术院（KAIST）等** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)）。

## 一句话定义

若硬件 **刻意编码多种非线性共振**，多步态可 **从力学涌现**，而非全靠高增益 PD 硬拧。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| NNM | Nonlinear Normal Mode | 非线性正规模 |
| SEA | Series Elastic Actuator | 串联弹性执行器 |
| COM | Center of Mass | 质心运动 |

## 为什么重要

动物用 **体态+刚度** 调共振换步态；多数四足机器人刚度高、步态由控制器单频决定。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 慕尼黑工业大学（TUM）、德国航空航天中心（DLR）、韩国科学技术院（KAIST）等 |
| **开源** | 见 [工程实践](#工程实践) |

## 核心原理

eBert：12 DoF，关节 SEA 刚度按小型犬标定；保守动力学求 **6 条 NNM**（平移/俯仰/滚转等耦合）；简单状态切换控制器 + 步长优化激发各模态为 **troting/pacing/bounding 类** 步态。

### 流程总览

```mermaid
flowchart LR
  design[柔顺腿 SEA 设计] --> nnm[识别 6 NNM]
  nnm --> opt[步长黑箱优化]
  opt --> gait[涌现多步态]
  gait --> hw[eBert 硬件验证]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-07** 无可运行官方代码（或本文为硬件/协议类工作）。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 见论文摘录与项目页核查结论 |
| 复现入口 | 以 arXiv 为准 |

## 实验与评测

仿真与硬件验证 NNM 存在性；各模态经优化后 **不同速度** 步态，硬件 **大体迁移**（见原文视频）。

## 结论

NNM 工具让 **非线性共振可设计、可预测**，为下一代 **具身智能四足** 提供 co-design 路线。

1. 宣称首个 **完整 3D 四足** 上系统应用 NNM 框架。
2. 刚度 **不可在线调**（相对生物肌肉）。
3. 最小反馈补偿摩擦。
4. 不是复刻动物能效数字。
5. **无开源**。

## 局限与风险

开环/弱反馈为主；工程鲁棒性与负载能力未对标工业四足。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [具身基础模型硬件共设计](../concepts/embodied-foundation-model-hardware-codesign.md)
- [paper-qlaun.md](./paper-qlaun.md)

## 参考来源

- [ebert_nonlinear_normal_modes_arxiv_2609_00539.md](../../sources/papers/ebert_nonlinear_normal_modes_arxiv_2609_00539.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.00539](https://arxiv.org/abs/2609.00539)
