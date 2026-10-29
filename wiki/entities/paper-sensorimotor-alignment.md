---
type: entity
project_id: sensorimotor-alignment
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-10-29
arxiv: "2610.31670"
project: https://yunfeixie233.github.io/s2/
code: https://github.com/yunfeixie233/sensorimotor-alignment
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-29-58-papers-technology-map.md
sources:
  - ../../sources/papers/sensorimotor_alignment_arxiv_2610_31670.md
  - ../../sources/repos/sensorimotor_alignment.md
  - ../../sources/sites/sensorimotor-alignment.md
  - ../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md
summary: "Sensorimotor Alignment（arXiv:2610.31670）：Sensorimotor Alignment比较感知表征与末端轨迹的几何结构，为昂贵闭环评测提供离线筛选信号；跨模型相关性值得参考，但不能替代真实任务成功率。"
---

# Sensorimotor Alignment（arXiv:2610.31670）

**Sensorimotor Alignment**（*Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success*，[arXiv:2610.31670](https://arxiv.org/abs/2610.31670)，[项目页](https://yunfeixie233.github.io/s2/)，[代码](https://github.com/yunfeixie233/sensorimotor-alignment)）来自 [具身智能小站 2026-10-29 盘点](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)，属「评测、安全与不确定性」组。

## 一句话定义

**Sensorimotor Alignment比较感知表征与末端轨迹的几何结构，为昂贵闭环评测提供离线筛选信号；跨模型相关性值得参考，但不能替代真实任务成功率。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sensorimotor Alignment | Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success | 本文方法 / 基准的简称 |

## 为什么重要

- 先用感知运动对齐筛选候选策略。
- 开源结论：**已开源**（步骤 2.5，2026-10-29）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.31670](https://arxiv.org/abs/2610.31670) |
| **开源** | **已开源** |
| **要点** | Sensorimotor Alignment比较感知表征与末端轨迹的几何结构，为昂贵闭环评测提供离线筛选信号；跨模型相关性值得参考，但不能替代真实任务成功率。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`yunfeixie233/sensorimotor-alignment`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as yunfeixie233/sensorimotor-alignment
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Sensorimotor Alignment 模型
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

- 横向索引见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**先用感知运动对齐筛选候选策略：Sensorimotor Alignment比较感知表征与末端轨迹的几何结构，为昂贵闭环评测提供离线筛选信号；跨模型相关性值得参考，但不能替代真实任务成功率。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：Sensorimotor Alignment比较感知表征与末端轨迹的几何结构，为昂贵闭环评测提供离线筛选信号；跨模型相关性值得参考，但不能替代真实任务成功率。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-29-58-papers-technology-map](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)

## 参考来源

- [sensorimotor_alignment_arxiv_2610_31670.md](../../sources/papers/sensorimotor_alignment_arxiv_2610_31670.md)
- [wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)
- [arXiv:2610.31670](https://arxiv.org/abs/2610.31670)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.31670)
- [项目页](https://yunfeixie233.github.io/s2/)
- [https://github.com/yunfeixie233/sensorimotor-alignment](https://github.com/yunfeixie233/sensorimotor-alignment)
