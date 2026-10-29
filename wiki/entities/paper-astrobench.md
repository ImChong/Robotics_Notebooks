---
type: entity
project_id: astrobench
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-10-29
arxiv: "2610.31064"
project: https://astroup.com/astrobench/
code: https://github.com/AstroUpLab/AstroBench
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-29-58-papers-technology-map.md
sources:
  - ../../sources/papers/astrobench_arxiv_2610_31064.md
  - ../../sources/repos/astrobench.md
  - ../../sources/sites/astrobench.md
  - ../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md
summary: "AstroBench（arXiv:2610.31064）：AstroBench区分物体运动条件、捕获终点与空间工位任务；通用策略直接迁移和专用ACT来自不同实验配置，应分别解释，不能将两者当成严格同条件排名。"
---

# AstroBench（arXiv:2610.31064）

**AstroBench**（*AstroBench: Benchmarking Free-Floating Capture and Space Robotic Manipulation*，[arXiv:2610.31064](https://arxiv.org/abs/2610.31064)，[项目页](https://astroup.com/astrobench/)，[代码](https://github.com/AstroUpLab/AstroBench)）来自 [具身智能小站 2026-10-29 盘点](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)，属「评测、安全与不确定性」组。

## 一句话定义

**AstroBench区分物体运动条件、捕获终点与空间工位任务；通用策略直接迁移和专用ACT来自不同实验配置，应分别解释，不能将两者当成严格同条件排名。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AstroBench | AstroBench: Benchmarking Free-Floating Capture and Space Robotic Manipulation | 本文方法 / 基准的简称 |

## 为什么重要

- 把自由漂浮抓取纳入空间操作评测。
- 开源结论：**已开源**（步骤 2.5，2026-10-29）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.31064](https://arxiv.org/abs/2610.31064) |
| **开源** | **已开源** |
| **要点** | AstroBench区分物体运动条件、捕获终点与空间工位任务；通用策略直接迁移和专用ACT来自不同实验配置，应分别解释，不能将两者当成严格同条件排名。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`AstroUpLab/AstroBench`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as AstroUpLab/AstroBench
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 AstroBench 模型
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

**把自由漂浮抓取纳入空间操作评测：AstroBench区分物体运动条件、捕获终点与空间工位任务；通用策略直接迁移和专用ACT来自不同实验配置，应分别解释，不能将两者当成严格同条件排名。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：AstroBench区分物体运动条件、捕获终点与空间工位任务；通用策略直接迁移和专用ACT来自不同实验配置，应分别解释，不能将两者当成严格同条件排名。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-29-58-papers-technology-map](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)

## 参考来源

- [astrobench_arxiv_2610_31064.md](../../sources/papers/astrobench_arxiv_2610_31064.md)
- [wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)
- [arXiv:2610.31064](https://arxiv.org/abs/2610.31064)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.31064)
- [项目页](https://astroup.com/astrobench/)
- [https://github.com/AstroUpLab/AstroBench](https://github.com/AstroUpLab/AstroBench)
