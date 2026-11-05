---
type: entity
project_id: hilo-wam
tags:
  - paper
  - world-model
  - control
status: complete
updated: 2026-11-05
arxiv: "2611.05023"
code: https://github.com/xywang227/HiLo-WAM
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/hilo_wam_arxiv_2611_05023.md
  - ../../sources/repos/hilo_wam.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "HiLo-WAM（arXiv:2611.05023）：HiLo-WAM 用稀疏画面跟踪任务进度，用密集画面观察眼前的手物接触，两路信息共同引导动作。"
---

# HiLo-WAM（arXiv:2611.05023）

**HiLo-WAM**（*HiLo-WAM: Decoupling Long-Horizon Task Evolution and Local Dynamics in World Action Models*，[arXiv:2611.05023](https://arxiv.org/abs/2611.05023)，[代码](https://github.com/xywang227/HiLo-WAM)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「世界模型与控制」组。

## 一句话定义

**HiLo-WAM 用稀疏画面跟踪任务进度，用密集画面观察眼前的手物接触，两路信息共同引导动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HiLo-WAM | HiLo-WAM: Decoupling Long-Horizon Task Evolution and Local Dynamics in World Action Models | 本文方法 / 基准的简称 |
| WAM | World Action Model | 世界–动作联合模型 |

## 为什么重要

- 用疏密两种画面预测长程双臂操作。作者在仿真里分别改变时间跨度和采样密度，也在真机上验证。
- 开源结论：**已开源**（步骤 2.5，2026-11-05）。仓库可达（git ls-remote）。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.05023](https://arxiv.org/abs/2611.05023) |
| **开源** | **已开源** |
| **要点** | HiLo-WAM 用稀疏画面跟踪任务进度，用密集画面观察眼前的手物接触，两路信息共同引导动作。 |
| **文内指标** | 公众号转述：作者在仿真里分别改变时间跨度和采样密度，也在真机上验证。 |

## 源码运行时序图

以下为按仓库入口（`xywang227/HiLo-WAM`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as xywang227/HiLo-WAM
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 HiLo-WAM 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：作者在仿真里分别改变时间跨度和采样密度，也在真机上验证。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**用疏密两种画面预测长程双臂操作：作者在仿真里分别改变时间跨度和采样密度，也在真机上验证。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：HiLo-WAM 用稀疏画面跟踪任务进度，用密集画面观察眼前的手物接触，两路信息共同引导动作。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [hilo_wam_arxiv_2611_05023.md](../../sources/papers/hilo_wam_arxiv_2611_05023.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.05023](https://arxiv.org/abs/2611.05023)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.05023)
- [https://github.com/xywang227/HiLo-WAM](https://github.com/xywang227/HiLo-WAM)
