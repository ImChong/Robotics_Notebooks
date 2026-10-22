---
type: entity
tags:
  - paper
  - hand-pose-estimation
  - mocap
  - calibration
  - dexterous-teleoperation
  - lora
  - hku
status: complete
updated: 2026-10-22
project_id: yoco
arxiv: "2610.11657"
project: https://franklin-zhang0.github.io/YOCO
code: https://github.com/Franklin-Zhang0/YOCO
related:
  - ../tasks/teleoperation.md
  - ../concepts/motion-retargeting.md
  - ./paper-adadex.md
sources:
  - ../../sources/papers/yoco_arxiv_2610_11657.md
summary: "YOCO（CoRL 2026，香港大学）：用 K=4 对样本经 HyperNet 预测冻结 MANO 的 LoRA 式更新，免微调地校准动捕手套与无标记手部追踪的个体偏差，并提升灵巧遥操作成功率。"
---

# YOCO（一次校准的动捕手部标定）

## 一句话定义

**YOCO 用少量成对的原始与目标手部姿态，经 HyperNet 一次前馈预测对冻结 MANO 估计器的 LoRA 式修正，使每个用户的动捕标定不再需要逐人微调。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MANO | MANO hand model | 参数化手部模型，本文保留其几何先验 |
| LoRA | Low-Rank Adaptation | 低秩适配，本文由 HyperNet 预测其更新 |
| PA-MPJPE | Procrustes-Aligned Mean Per-Joint Position Error | 对齐后的关节位置误差，本文主要指标 |
| HyperNet | Hypernetwork | 生成另一网络参数的网络，本文用于预测 LoRA 更新 |
| AVP | Apple Vision Pro | 作为真实手套评测中的参考来源（据项目页） |

## 为什么重要

- 低成本手套与无标记追踪器的个体偏差会直接传到灵巧遥操作与数据采集中，逐人标定耗时。
- 把标定做成少样本、免微调的一次前馈，使其更容易用于大规模遥操作数据采集（推测）。

## 方法与机制

- **输入：** K = 4 组成对的原始手部姿态与目标姿态。
- **HyperNet：** 以样本为条件，预测对冻结 MANO 估计器的 LoRA 式更新。
- **训练：** 在 InterHand2.6M 上加入合成的漂移增强。

## 实验与评测

- 增强 InterHand2.6M（K = 4）：PA-MPJPE 16.55 → 10.45 mm，优于 LoRA-Finetune 的 11.67 mm。
- 真实手套对 AVP：PA-MPJPE 14.63 → 13.14 mm，LoRA-Finetune 为 13.94 mm。
- 灵巧遥操作（4 任务，40 次）：成功率 90% 对比 LoRA-Finetune 65% 与未校准 35%。

## 与其他工作对比

- 主要对照 LoRA-Finetune（逐人微调）与未校准输入。
- 项目页说明，遥操作对比使用各自独立训练的重定向模型，增益不能全部归因于校准本身。

## 结论

本文的核心判断是：手部追踪的个体偏差可以用少量样本经超网络一次修正，且保留 MANO 的先验比逐人微调更适合少样本场景。

- K = 4 的少样本校准即可在多项指标上优于逐人微调基线。
- 遥操作成功率提升明显，但该对比包含不同的重定向模型，结论需谨慎。
- 证据来自项目页数字与摘要；代码与模型的可用性未在本次核实。

## 工程实践

- 代码与模型：项目页列出 GitHub 与 Hugging Face 合集；本次未访问。
- 复现需要：MANO 估计器、InterHand2.6M 训练数据与真实手套采集数据。

## 局限与风险

- 摘要未给出数字，数值来自项目页，需回到论文全文核对。
- 遥操作对比未隔离校准的单独贡献（项目页自述）。
- 推测：标定样本的质量与覆盖范围会限制未见姿态上的泛化。

## 关联页面

- [Teleoperation（遥操作）](../tasks/teleoperation.md) — 本文的应用场景
- [Motion Retargeting](../concepts/motion-retargeting.md) — 遥操作中人手到机器人的适配环节
- [AdaDex](./paper-adadex.md) — 同样面向灵巧手与人手轨迹的相近工作

## 参考来源

- [YOCO 论文来源归档](../../sources/papers/yoco_arxiv_2610_11657.md)

## 推荐继续阅读

- [arXiv 2610.11657](https://arxiv.org/abs/2610.11657)
- [项目页](https://franklin-zhang0.github.io/YOCO)
