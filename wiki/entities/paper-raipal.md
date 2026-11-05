---
type: entity
project_id: raipal
tags:
  - paper
  - world-model
  - control
status: complete
updated: 2026-11-05
arxiv: "2611.04334"
code: https://github.com/donghyuk1/RAIPAL_SOEM
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/locomotion.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/raipal_arxiv_2611_04334.md
  - ../../sources/repos/raipal.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "RAIPAL（arXiv:2611.04334）：RAIPAL 依据人臂执行任务时的力量、速度和耐久需求设计关节。"
---

# RAIPAL（arXiv:2611.04334）

**RAIPAL**（*RAIPAL: A Powerful and Agile Humanoid Arm for Human Tasks*，[arXiv:2611.04334](https://arxiv.org/abs/2611.04334)，[代码](https://github.com/donghyuk1/RAIPAL_SOEM)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「世界模型与控制」组。

## 一句话定义

**RAIPAL 依据人臂执行任务时的力量、速度和耐久需求设计关节。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RAIPAL | RAIPAL: A Powerful and Agile Humanoid Arm for Human Tasks | 本文方法 / 基准的简称 |

## 为什么重要

- 机械臂怎样兼顾提重物和快速挥动。原型在不同测试中分别达到 20 公斤峰值负载和每秒 20.6 米的末端速度，持续负载另做热分析。
- 开源结论：**已开源**（步骤 2.5，2026-11-05）。仓库可达（git ls-remote）。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.04334](https://arxiv.org/abs/2611.04334) |
| **开源** | **已开源** |
| **要点** | RAIPAL 依据人臂执行任务时的力量、速度和耐久需求设计关节。 |
| **文内指标** | 公众号转述：原型在不同测试中分别达到 20 公斤峰值负载和每秒 20.6 米的末端速度，持续负载另做热分析。 |

## 源码运行时序图

以下为按仓库入口（`donghyuk1/RAIPAL_SOEM`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as donghyuk1/RAIPAL_SOEM
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 RAIPAL 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：原型在不同测试中分别达到 20 公斤峰值负载和每秒 20.6 米的末端速度，持续负载另做热分析。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**机械臂怎样兼顾提重物和快速挥动：原型在不同测试中分别达到 20 公斤峰值负载和每秒 20.6 米的末端速度，持续负载另做热分析。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：RAIPAL 依据人臂执行任务时的力量、速度和耐久需求设计关节。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [locomotion](../tasks/locomotion.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [raipal_arxiv_2611_04334.md](../../sources/papers/raipal_arxiv_2611_04334.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.04334](https://arxiv.org/abs/2611.04334)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.04334)
- [https://github.com/donghyuk1/RAIPAL_SOEM](https://github.com/donghyuk1/RAIPAL_SOEM)
