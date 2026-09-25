---
type: entity
tags: [paper, dataset, deformable, bimanual, fudan, cmu]
status: complete
updated: 2026-09-25
arxiv: "2609.10243"
related:
  - ../tasks/bimanual-manipulation.md
  - ./paper-crvae-deformable-manipulation-partial-obs.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/foldex_deformable_benchmark_arxiv_2609_10243.md
summary: "FolDeX（arXiv:2609.10243）：2000+ hours real data, 20+ tasks, 10+ robots; FoldChallenge; FoldScore metric；截至入库日未见官方代码。"
---

# FolDeX（arXiv:2609.10243）

**FolDeX**（*FolDeX: A Physical-World Benchmark for Long-Horizon Robotic Manipulation of Deformable Objects*，[arXiv:2609.10243](https://arxiv.org/abs/2609.10243)）由 **复旦大学（Fudan）；美的（Midea）；卡内基梅隆大学（CMU）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_manipulation_2026-09-14.md)）。

## 一句话定义

FolDeX：面向可变形物体长程机器人操作的真实世界基准 — 2000+ hours real data, 20+ tasks, 10+ robots。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FolDeX | Fold Deformable eX benchmark | 本文基准 |
| FoldScore | FoldScore Metric | 折叠质量评分 |
| BC | Behavior Cloning | 行为克隆基线 |

## 为什么重要

可变形衣物长程操作缺大规模真实基准；仿真到真差距大。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 复旦大学（Fudan）；美的（Midea）；卡内基梅隆大学（CMU） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

### 数据集速查四维

| 维度 | 归档口径（2026-09-14） |
|------|----------------------|
| **规模** | 2000+ 小时真实采集；20+ 任务；10+ 机器人本体 |
| **模态** | **归档未落明细** — 摘录只给出「物理世界真实采集」与本体数量，未列出具体传感通道（视频 / 深度 / 点云 / 触觉 / 本体状态各有无、分辨率与频率均未知）。代码与数据卡发布后须回填本行，**勿按同类衣物数据集的惯例默认** |
| **许可证** | **未知** — 无官方仓库，也就没有 LICENSE 与使用条款；商用与再分发边界均待确认 |
| **重定向就绪度** | 10+ 机器人形态混采，跨本体复用须先做观测与动作空间归一化（见「工程实践」）；归档未说明是否提供统一的重定向脚本或标准骨架 |

- **读法：** 四维中只有 **规模** 一项有硬数字。把本页当成「知道有这么个基准」的索引，而不是可直接选型的数据卡。

## 核心原理

采集 2000+ 小时多机器人双手机物数据；定义 FoldChallenge 任务集与 FoldScore；供算法横向对比。

### 流程总览

```mermaid
flowchart LR
  collect[多机真实采集] --> data[2000+ h 数据]
  data --> tasks[20+ 任务]
  tasks --> fold[FoldChallenge]
  fold --> score[FoldScore]
```

## 源码运行时序图

**不适用** — 本文为理论/硬件/系统/数据类工作，arXiv 未提供可运行训练或部署仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 使用 benchmark 需遵循 FoldScore 协议；多机器人形态差异需归一化观测。 |

## 实验与评测

20+ 任务；FoldScore 与现有 SOTA 对比（见论文）。

## 结论

FolDeX 用千小时级真实数据与 FoldScore 建立可变形衣物长程操作基准。

1. 2000+ 小时真实规模罕见。
2. 10+ 机器人提升多样性。
3. FoldScore 量化折叠质量。
4. 长程比单步折叠更难。
5. 数据发布状态见开源核查。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 仿真布料 benchmark | 物理差距大 |
| 短 horizon 折叠集 | 不测长程 |

## 局限与风险

数据获取成本高；衣物类别仍有限。

## 关联页面

- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [./paper-crvae-deformable-manipulation-partial-obs.md](./paper-crvae-deformable-manipulation-partial-obs.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [foldex_deformable_benchmark_arxiv_2609_10243.md](../../sources/papers/foldex_deformable_benchmark_arxiv_2609_10243.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_manipulation_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.10243](https://arxiv.org/abs/2609.10243)
