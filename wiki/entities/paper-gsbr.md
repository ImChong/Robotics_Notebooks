---
type: entity
project_id: gsbr
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-05
arxiv: "2611.05295"
project: https://charlesxu0124.github.io/gsbr/
code: https://github.com/charlesxu0124/gsbr
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/gsbr_arxiv_2611_05295.md
  - ../../sources/repos/gsbr.md
  - ../../sources/sites/gsbr.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "GSBR（arXiv:2611.05295）：GSBR 用基础策略采样的动作估计可行范围，范围内允许策略挑出表现更好的动作，越过边界才逐渐加重惩罚。"
---

# GSBR（arXiv:2611.05295）

**GSBR**（*GSBR: Geometric Support Behavior Regularization for Online RL*，[arXiv:2611.05295](https://arxiv.org/abs/2611.05295)，[项目页](https://charlesxu0124.github.io/gsbr/)，[代码](https://github.com/charlesxu0124/gsbr)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「策略泛化与评测」组。

## 一句话定义

**GSBR 用基础策略采样的动作估计可行范围，范围内允许策略挑出表现更好的动作，越过边界才逐渐加重惩罚。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GSBR | GSBR: Geometric Support Behavior Regularization for Online RL | 本文方法 / 基准的简称 |

## 为什么重要

- 在线强化学习应离示范动作多远。作者在四个机器人仿真基准上比较了这种做法与硬性限制。
- 开源结论：**已开源**（步骤 2.5，2026-11-05）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.05295](https://arxiv.org/abs/2611.05295) |
| **开源** | **已开源** |
| **要点** | GSBR 用基础策略采样的动作估计可行范围，范围内允许策略挑出表现更好的动作，越过边界才逐渐加重惩罚。 |
| **文内指标** | 公众号转述：作者在四个机器人仿真基准上比较了这种做法与硬性限制。 |

## 源码运行时序图

以下为按仓库入口（`charlesxu0124/gsbr`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as charlesxu0124/gsbr
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 GSBR 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：作者在四个机器人仿真基准上比较了这种做法与硬性限制。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**在线强化学习应离示范动作多远：作者在四个机器人仿真基准上比较了这种做法与硬性限制。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：GSBR 用基础策略采样的动作估计可行范围，范围内允许策略挑出表现更好的动作，越过边界才逐渐加重惩罚。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [gsbr_arxiv_2611_05295.md](../../sources/papers/gsbr_arxiv_2611_05295.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.05295](https://arxiv.org/abs/2611.05295)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.05295)
- [项目页](https://charlesxu0124.github.io/gsbr/)
- [https://github.com/charlesxu0124/gsbr](https://github.com/charlesxu0124/gsbr)
