---
type: entity
project_id: gpu-incremental-edt
tags:
  - paper
  - planning
  - optimization
status: complete
updated: 2026-11-03
arxiv: "2610.41667"
code: https://github.com/JINXER000/GIE-mapping
related:
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../overview/embodied-station-2026-11-03-49-papers-technology-map.md
sources:
  - ../../sources/papers/gpu_incremental_edt_arxiv_2610_41667.md
  - ../../sources/repos/gpu_incremental_edt.md
  - ../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md
summary: "GPU Incremental EDT（arXiv:2610.41667）：通过 GPU 并行计算局部距离信息，并将新观测融合进全局距离场，服务移动机器人规划。"
---

# GPU Incremental EDT（arXiv:2610.41667）

**GPU Incremental EDT**（*GPU-accelerated Incremental Euclidean Distance Transform for Online Motion Planning of Mobile Robots*，[arXiv:2610.41667](https://arxiv.org/abs/2610.41667)，[代码](https://github.com/JINXER000/GIE-mapping)）来自 [具身智能小站 2026-11-03 盘点](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)，属「规划、优化与感知基础」组。

## 一句话定义

**通过 GPU 并行计算局部距离信息，并将新观测融合进全局距离场，服务移动机器人规划。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU Incremental EDT | GPU-accelerated Incremental Euclidean Distance Transform for Online Motion Planning of Mobile Robots | 本文方法 / 基准的简称 |

## 为什么重要

- 为在线规划增量更新距离场。正文标明 2022 年接收，应作为历史工程方法阅读。
- 开源结论：**已开源**（步骤 2.5，2026-11-03）。仓库可达（git ls-remote）。
- 同期同组工作见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.41667](https://arxiv.org/abs/2610.41667) |
| **开源** | **已开源** |
| **要点** | 通过 GPU 并行计算局部距离信息，并将新观测融合进全局距离场，服务移动机器人规划。 |
| **文内指标** | 公众号转述：正文标明 2022 年接收，应作为历史工程方法阅读。 |

## 源码运行时序图

以下为按仓库入口（`JINXER000/GIE-mapping`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as JINXER000/GIE-mapping
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 GPU Incremental EDT 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：正文标明 2022 年接收，应作为历史工程方法阅读。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**为在线规划增量更新距离场：正文标明 2022 年接收，应作为历史工程方法阅读。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：通过 GPU 并行计算局部距离信息，并将新观测融合进全局距离场，服务移动机器人规划。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [locomotion](../tasks/locomotion.md)
- [sim2real](../concepts/sim2real.md)
- [embodied-station-2026-11-03-49-papers-technology-map](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)

## 参考来源

- [gpu_incremental_edt_arxiv_2610_41667.md](../../sources/papers/gpu_incremental_edt_arxiv_2610_41667.md)
- [wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)
- [arXiv:2610.41667](https://arxiv.org/abs/2610.41667)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.41667)
- [https://github.com/JINXER000/GIE-mapping](https://github.com/JINXER000/GIE-mapping)
