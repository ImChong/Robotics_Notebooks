---
type: entity
project_id: escape
tags:
  - paper
  - manipulation
  - contact-rich
status: complete
updated: 2026-11-01
arxiv: "2610.23943"
project: https://escape-policy.github.io/
code: https://github.com/weiqingshen/ESCAPE
related:
  - ../tasks/manipulation.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/tactile-sensing.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/escape_arxiv_2610_23943.md
  - ../../sources/repos/escape.md
  - ../../sources/sites/escape.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "ESCAPE（arXiv:2610.23943）：ESCAPE聚合分布式近场深度，以流匹配生成避障轨迹，仿真训练后迁移真机；适合检查外部相机依赖，当前实验主要是单臂静态杂乱环境到达。"
---

# ESCAPE（arXiv:2610.23943）

**ESCAPE**（*ESCAPE: Egocentric Sensing for Collision Avoidance and Policy Execution*，[arXiv:2610.23943](https://arxiv.org/abs/2610.23943)，[项目页](https://escape-policy.github.io/)，[代码](https://github.com/weiqingshen/ESCAPE)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「灵巧、接触与精细操作」组。

## 一句话定义

**ESCAPE聚合分布式近场深度，以流匹配生成避障轨迹，仿真训练后迁移真机；适合检查外部相机依赖，当前实验主要是单臂静态杂乱环境到达。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ESCAPE | ESCAPE: Egocentric Sensing for Collision Avoidance and Policy Execution | 本文方法 / 基准的简称 |

## 为什么重要

- 用机械臂自身的深度感知绕开遮挡。
- 开源结论：**已开源**（步骤 2.5，2026-11-01）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.23943](https://arxiv.org/abs/2610.23943) |
| **开源** | **已开源** |
| **要点** | ESCAPE聚合分布式近场深度，以流匹配生成避障轨迹，仿真训练后迁移真机；适合检查外部相机依赖，当前实验主要是单臂静态杂乱环境到达。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`weiqingshen/ESCAPE`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as weiqingshen/ESCAPE
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 ESCAPE 模型
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

**用机械臂自身的深度感知绕开遮挡：ESCAPE聚合分布式近场深度，以流匹配生成避障轨迹，仿真训练后迁移真机；适合检查外部相机依赖，当前实验主要是单臂静态杂乱环境到达。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：ESCAPE聚合分布式近场深度，以流匹配生成避障轨迹，仿真训练后迁移真机；适合检查外部相机依赖，当前实验主要是单臂静态杂乱环境到达。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [contact-rich-manipulation](../concepts/contact-rich-manipulation.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [escape_arxiv_2610_23943.md](../../sources/papers/escape_arxiv_2610_23943.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.23943](https://arxiv.org/abs/2610.23943)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.23943)
- [项目页](https://escape-policy.github.io/)
- [https://github.com/weiqingshen/ESCAPE](https://github.com/weiqingshen/ESCAPE)
