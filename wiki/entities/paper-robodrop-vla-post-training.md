---
type: entity
tags: [paper, vla, post-training, data-curation, tsinghua]
status: complete
updated: 2026-09-15
arxiv: "2609.10021"
related:
  - ../tasks/manipulation.md
  - ./paper-openwam.md
  - ./paper-vla-precision.md
sources:
  - ../../sources/papers/robodrop_vla_post_training_arxiv_2609_10021.md
summary: "RoboDrop（arXiv:2609.10021）：gradient compatibility vs semantic/visual val samples; real robot SR 35%→67.5%；截至入库日未见官方代码。"
---

# RoboDrop（arXiv:2609.10021）

**RoboDrop**（*RoboDrop: Curating VLA Post-Training Data via Local Gradient Compatibility*，[arXiv:2609.10021](https://arxiv.org/abs/2609.10021)）由 **清华大学（Tsinghua）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_manipulation_2026-09-14.md)）。

## 一句话定义

RoboDrop：基于局部梯度兼容性的 VLA 后训练数据筛选 — gradient compatibility vs semantic/visual val samples。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作 |
| SR | Success Rate | 任务成功率 |
| SFT | Supervised Fine-Tuning | 监督微调 |

## 为什么重要

VLA 后训练数据嘈杂；语义相似样本可能损害梯度方向。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 清华大学（Tsinghua） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

计算候选样本与当前策略的 local gradient compatibility；丢弃不兼容样本；优于仅用 semantic/visual validation 选数据。

### 流程总览

```mermaid
flowchart LR
  pool[后训练数据池] --> grad[梯度兼容性评分]
  grad --> keep[保留高兼容样本]
  keep --> sft[VLA 微调]
  sft --> robot[真机部署]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-14** arXiv 与常见项目页 **未见** 官方可运行代码仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 兼容性阈值需按任务调；计算梯度开销可用小批量近似。 |

## 实验与评测

真机成功率 35%→67.5%；与 random/semantic 筛选对比。

## 结论

RoboDrop 用梯度兼容性策展 VLA 后训练数据，真机成功率近乎翻倍。

1. 语义相似≠梯度有益。
2. 局部兼容性可自动筛毒样本。
3. SR 35%→67.5%。
4. 适合小数据真机微调。
5. 需访问策略梯度。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 全量后训练 | 噪声拖累性能 |
| 语义去重 | 不保证梯度一致 |

## 局限与风险

梯度计算成本；对极大数据池需近似。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [./paper-openwam.md](./paper-openwam.md)
- [./paper-vla-precision.md](./paper-vla-precision.md)

## 参考来源

- [robodrop_vla_post_training_arxiv_2609_10021.md](../../sources/papers/robodrop_vla_post_training_arxiv_2609_10021.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_manipulation_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.10021](https://arxiv.org/abs/2609.10021)
