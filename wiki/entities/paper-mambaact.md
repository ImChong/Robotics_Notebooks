---
type: entity
project_id: mambaact
tags:
  - paper
  - vla
  - policy-learning
status: complete
updated: 2026-11-01
arxiv: "2610.22951"
code: https://github.com/giakhuyendihoc/MambaAct
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/mambaact_arxiv_2610_22951.md
  - ../../sources/repos/mambaact.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "MambaAct（arXiv:2610.22951）：MambaAct保留视觉语言骨干，以紧凑Mamba专家和上下文压缩生成动作；多种子仿真与真机显示延迟收益，去噪部分的加速不应等同整机控制加速。"
---

# MambaAct（arXiv:2610.22951）

**MambaAct**（*MambaAct: Efficient Mamba Action Experts for Flow-Matching Vision-Language-Action Models*，[arXiv:2610.22951](https://arxiv.org/abs/2610.22951)，[代码](https://github.com/giakhuyendihoc/MambaAct)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「VLA 与策略学习」组。

## 一句话定义

**MambaAct保留视觉语言骨干，以紧凑Mamba专家和上下文压缩生成动作；多种子仿真与真机显示延迟收益，去噪部分的加速不应等同整机控制加速。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MambaAct | MambaAct: Efficient Mamba Action Experts for Flow-Matching Vision-Language-Action Models | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 压缩重复去噪的动作专家。
- 开源结论：**已开源**（步骤 2.5，2026-11-01）。仓库可达（git ls-remote）。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.22951](https://arxiv.org/abs/2610.22951) |
| **开源** | **已开源** |
| **要点** | MambaAct保留视觉语言骨干，以紧凑Mamba专家和上下文压缩生成动作；多种子仿真与真机显示延迟收益，去噪部分的加速不应等同整机控制加速。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`giakhuyendihoc/MambaAct`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as giakhuyendihoc/MambaAct
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 MambaAct 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**压缩重复去噪的动作专家：MambaAct保留视觉语言骨干，以紧凑Mamba专家和上下文压缩生成动作；多种子仿真与真机显示延迟收益，去噪部分的加速不应等同整机控制加速。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：MambaAct保留视觉语言骨干，以紧凑Mamba专家和上下文压缩生成动作；多种子仿真与真机显示延迟收益，去噪部分的加速不应等同整机控制加速。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [mambaact_arxiv_2610_22951.md](../../sources/papers/mambaact_arxiv_2610_22951.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.22951](https://arxiv.org/abs/2610.22951)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.22951)
- [https://github.com/giakhuyendihoc/MambaAct](https://github.com/giakhuyendihoc/MambaAct)
