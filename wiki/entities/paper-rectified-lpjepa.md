---
type: entity
tags: [paper, jepa, sparse-representation, distribution-matching, representation-learning, icml]
project_id: rectified-lpjepa
status: complete
updated: 2026-10-20
arxiv: "2602.01456"
venue: "ICML 2026"
related:
  - ./paper-lpwm.md
  - ./paper-lewm.md
  - ./paper-levjepa.md
  - ./paper-lejepa.md
sources:
  - ../../sources/papers/rectified_lpjepa_arxiv_2602_01456.md
  - ../../sources/blogs/wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md
summary: "Rectified LpJEPA（arXiv:2602.01456，ICML 2026）：用 RDMReg 把 JEPA 表示分布匹配到 Rectified Generalized Gaussian，得到可控稀疏的非负表示；arXiv 页未列代码。"
---

# Rectified LpJEPA（可控稀疏的 JEPA 表示）

**Rectified LpJEPA** 在 JEPA 的分布匹配正则里，把原本的各向同性高斯换成 Rectified Generalized Gaussian，使表示可以显式控制稀疏程度，并保持非负。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| JEPA | Joint-Embedding Predictive Architecture | 通过预测潜表示学习的架构 |
| RDMReg | Rectified Distribution Matching Regularization | 本文提出的分布匹配正则，基于切片双样本损失 |
| LeJEPA | — | 以各向同性高斯做防坍缩的前序工作（见关联页） |
| ℓ0 | ℓ0 norm | 非零元素个数，本文用其期望控制稀疏度 |

## 为什么重要

稠密表示中大部分维度都会被激活，难以对应「少数相关因素决定世界」这一直觉。Rectified LpJEPA 证明稀疏、非负表示也能保持竞争力，为后续 [LpWM](./paper-lpwm.md) 把稀疏表示放进世界模型提供基础。

## 核心信息

- 作者：Yilun Kuang、Yash Dagade、Tim G. J. Rudner、Randall Balestriero、Yann LeCun（arXiv 页）
- 机构：arXiv 页未显示
- 代码：arXiv 页未列官方仓库

## 核心原理（方法栈）

1. **JEPA 基础**：视图不变的预测表示。
2. **分布匹配**：用切片双样本损失把表示分布对齐到目标分布。
3. **目标分布**：Rectified Generalized Gaussian，可控制期望 ℓ0 范数。
4. **结果**：非负、可控稀疏的表示。

## 实验与评测

- 摘要称：稀疏性与性能之间有良好权衡，图像分类结果有竞争力。
- 具体数据集与数值未在本次入库中核对原表。

## 工程实践

- 无官方代码，复现需按 RDMReg 定义自行实现。
- 稀疏度是可调超参，应同时报告稀疏度与下游指标，避免只看一方。

## 局限与风险

- 证据以图像分类为主，尚未覆盖机器人控制（本页未发现相关实验描述）。
- 无代码，难以确认切片双样本损失的实现细节。

## 与其他工作对比

| 维度 | Rectified LpJEPA | [LeVJEPA](./paper-levjepa.md) |
|------|------------------|-------------------------------|
| 目标分布 | 非负稀疏 Rectified GG | 各向同性高斯（SIGReg 一类） |
| 表示形态 | 稀疏 | 稠密 |

## 结论

**稀疏、非负表示可以作为 JEPA 的合法目标，且稀疏度可控。**

1. **目标分布可替换** — 把高斯换成 Rectified GG 仍能防止坍缩（摘要结论）。
2. **稀疏度可调** — 用 ℓ0 期望控制，便于在效率与信息量间取舍。
3. **证据范围** — 主要是图像任务，机器人场景需单独验证。
4. **衔接** — 是 [LpWM](./paper-lpwm.md) 稀疏世界模型的理论与方法前序。

## 关联页面

- [LpWM](./paper-lpwm.md) — 把稀疏表示用于 action-conditioned 世界模型
- [LeWorldModel](./paper-lewm.md) — 稠密高斯表示的对照
- [LeVJEPA](./paper-levjepa.md) — 同属防坍缩的 JEPA 正则改进
- [LeJEPA](./paper-lejepa.md) — 高斯分布匹配的前序工作

## 参考来源

- [Rectified LpJEPA 论文归档](../../sources/papers/rectified_lpjepa_arxiv_2602_01456.md)
- [深蓝具身智能文章归档](../../sources/blogs/wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md)

## 推荐继续阅读

- [arXiv:2602.01456](https://arxiv.org/abs/2602.01456)
