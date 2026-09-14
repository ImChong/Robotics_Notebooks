---
type: entity
tags: [paper, vla, action-chunking, manipulation]
status: complete
updated: 2026-09-14
arxiv: "2609.09925"
related:
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
  - ./paper-openwam.md
sources:
  - ../../sources/papers/tfgca_chunked_vla_arxiv_2609_09925.md
summary: "TFGCA（arXiv:2609.09925）：TFGCA module; stationary wavelet + geometric cross-attention; zero-init residual; LIBERO-Plus +6.3, RoboTwin +28.5, AgiB；截至入库日未见官方代码。"
---

# TFGCA（arXiv:2609.09925）

**TFGCA**（*Time-Frequency Geometric Cross-Attention for Chunked Vision-Language-Action Models*，[arXiv:2609.09925](https://arxiv.org/abs/2609.09925)）由 **（论文未在 digest 标注机构）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_manipulation_2026-09-14.md)）。

## 一句话定义

面向分块视觉-语言-动作模型的时频几何交叉注意力 — TFGCA module。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作模型 |
| TFGCA | Time-Frequency Geometric Cross-Attention | 本文注意力模块 |
| DWT | Discrete Wavelet Transform | 离散小波变换 |

## 为什么重要

动作分块 VLA 需同时捕捉时域与频域动态；纯时域注意力易丢周期性结构。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | （论文未在 digest 标注机构） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

平稳小波分解动作/视觉特征；几何 cross-attention 对齐时空；zero-init residual 叠加到基座 VLA。

### 流程总览

```mermaid
flowchart LR
  vla[Chunked VLA] --> swt[平稳小波]
  swt --> tgca[TFGCA]
  tgca --> chunk[动作块输出]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-14** arXiv 与常见项目页 **未见** 官方可运行代码仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 小波层数与 chunk 长度需对齐；零初始化避免破坏预训练。 |

## 实验与评测

LIBERO-Plus +6.3；RoboTwin +28.5；AgiBot A2 +11.67pp。

## 结论

TFGCA 在时频几何域增强 chunked VLA，多基准一致上涨。

1. 时频分解适合周期性操作。
2. 几何 cross-attention 对齐空间结构。
3. 零初始化 residual 保稳定。
4. LIBERO-Plus/RoboTwin 涨幅大。
5. 模块可插拔基座 VLA。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 纯时域 chunked VLA | 丢频域结构 |
| 无 cross-attn 融合 | 模态对齐弱 |

## 局限与风险

机构未标注；真机仅部分基准；算力开销增加。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [./paper-openwam.md](./paper-openwam.md)

## 参考来源

- [tfgca_chunked_vla_arxiv_2609_09925.md](../../sources/papers/tfgca_chunked_vla_arxiv_2609_09925.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_manipulation_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.09925](https://arxiv.org/abs/2609.09925)
