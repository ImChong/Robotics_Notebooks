---
type: entity
project_id: havana
tags:
  - paper
  - embodied-agent
  - navigation
status: complete
updated: 2026-11-01
arxiv: "2610.24122"
code: https://github.com/Anastasiawd/HAVANA
related:
  - ../concepts/llm-robotics-control-interfaces.md
  - ../tasks/vision-language-navigation.md
  - ../tasks/manipulation.md
  - ../tasks/vision-language-navigation.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/havana_arxiv_2610_24122.md
  - ../../sources/repos/havana.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "HAVANA（arXiv:2610.24122）：HAVANA把语言条件潜计划与反应式局部轨迹分开，异步运行以减少重复预测；导航基准及不同形态实机展示效果，推理阶段无需生成未来视频。"
---

# HAVANA（arXiv:2610.24122）

**HAVANA**（*HAVANA: A Hierarchical Video-Action Model for Vision-Language Navigation*，[arXiv:2610.24122](https://arxiv.org/abs/2610.24122)，[代码](https://github.com/Anastasiawd/HAVANA)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「具身智能体与导航」组。

## 一句话定义

**HAVANA把语言条件潜计划与反应式局部轨迹分开，异步运行以减少重复预测；导航基准及不同形态实机展示效果，推理阶段无需生成未来视频。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HAVANA | HAVANA: A Hierarchical Video-Action Model for Vision-Language Navigation | 本文方法 / 基准的简称 |

## 为什么重要

- 慢速想路线，快速跟上最新观测。
- 开源结论：**已开源**（步骤 2.5，2026-11-01）。仓库可达（git ls-remote）。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.24122](https://arxiv.org/abs/2610.24122) |
| **开源** | **已开源** |
| **要点** | HAVANA把语言条件潜计划与反应式局部轨迹分开，异步运行以减少重复预测；导航基准及不同形态实机展示效果，推理阶段无需生成未来视频。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`Anastasiawd/HAVANA`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as Anastasiawd/HAVANA
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 HAVANA 模型
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

**慢速想路线，快速跟上最新观测：HAVANA把语言条件潜计划与反应式局部轨迹分开，异步运行以减少重复预测；导航基准及不同形态实机展示效果，推理阶段无需生成未来视频。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：HAVANA把语言条件潜计划与反应式局部轨迹分开，异步运行以减少重复预测；导航基准及不同形态实机展示效果，推理阶段无需生成未来视频。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [manipulation](../tasks/manipulation.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [havana_arxiv_2610_24122.md](../../sources/papers/havana_arxiv_2610_24122.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.24122](https://arxiv.org/abs/2610.24122)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.24122)
- [https://github.com/Anastasiawd/HAVANA](https://github.com/Anastasiawd/HAVANA)
