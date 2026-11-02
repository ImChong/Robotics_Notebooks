---
type: entity
project_id: nestedvla
tags:
  - paper
  - vla
  - policy-learning
status: complete
updated: 2026-11-01
arxiv: "2610.23271"
code: https://github.com/WanyuGroup/NeurIPS2026-NestedVLA
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/nestedvla_arxiv_2610_23271.md
  - ../../sources/repos/nestedvla.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "NestedVLA（arXiv:2610.23271）：NestedVLA用超网络生成动作专家参数，并单独保存压缩的视觉语言残差；三个仿真基准支持减少技能冲突与存储，尚不能视为任意新技能的零样本组合。"
---

# NestedVLA（arXiv:2610.23271）

**NestedVLA**（*NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models*，[arXiv:2610.23271](https://arxiv.org/abs/2610.23271)，[代码](https://github.com/WanyuGroup/NeurIPS2026-NestedVLA)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「VLA 与策略学习」组。

## 一句话定义

**NestedVLA用超网络生成动作专家参数，并单独保存压缩的视觉语言残差；三个仿真基准支持减少技能冲突与存储，尚不能视为任意新技能的零样本组合。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| NestedVLA | NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 不直接合并权重，而是按任务生成技能。
- 开源结论：**已开源**（步骤 2.5，2026-11-01）。仓库可达（git ls-remote）。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.23271](https://arxiv.org/abs/2610.23271) |
| **开源** | **已开源** |
| **要点** | NestedVLA用超网络生成动作专家参数，并单独保存压缩的视觉语言残差；三个仿真基准支持减少技能冲突与存储，尚不能视为任意新技能的零样本组合。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`WanyuGroup/NeurIPS2026-NestedVLA`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as WanyuGroup/NeurIPS2026-NestedVLA
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 NestedVLA 模型
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

**不直接合并权重，而是按任务生成技能：NestedVLA用超网络生成动作专家参数，并单独保存压缩的视觉语言残差；三个仿真基准支持减少技能冲突与存储，尚不能视为任意新技能的零样本组合。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：NestedVLA用超网络生成动作专家参数，并单独保存压缩的视觉语言残差；三个仿真基准支持减少技能冲突与存储，尚不能视为任意新技能的零样本组合。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [nestedvla_arxiv_2610_23271.md](../../sources/papers/nestedvla_arxiv_2610_23271.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.23271](https://arxiv.org/abs/2610.23271)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.23271)
- [https://github.com/WanyuGroup/NeurIPS2026-NestedVLA](https://github.com/WanyuGroup/NeurIPS2026-NestedVLA)
