---
type: entity
project_id: vaguebench
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-04
arxiv: "2611.03765"
code: https://github.com/stardreaming/VagueBench
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/vaguebench_arxiv_2611_03765.md
  - ../../sources/repos/vaguebench.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "VagueBench（arXiv:2611.03765）：删除指令中非必要的背景提示，同时保持任务语义不变，测试上下文依赖；以成对指令检查规划与执行退化，提醒评测不要只覆盖冗长明确的表达。"
---

# VagueBench（arXiv:2611.03765）

**VagueBench**（*VagueBench: Benchmarking the Robustness of Embodied Agents on Ambient Context Illusions*，[arXiv:2611.03765](https://arxiv.org/abs/2611.03765)，[代码](https://github.com/stardreaming/VagueBench)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「评测、数据与基准」组。

## 一句话定义

**删除指令中非必要的背景提示，同时保持任务语义不变，测试上下文依赖；以成对指令检查规划与执行退化，提醒评测不要只覆盖冗长明确的表达。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VagueBench | VagueBench: Benchmarking the Robustness of Embodied Agents on Ambient Context Illusions | 本文方法 / 基准的简称 |

## 为什么重要

- 指令少说几句，机器人还会做吗。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。仓库可达（git ls-remote）。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.03765](https://arxiv.org/abs/2611.03765) |
| **开源** | **已开源** |
| **要点** | 删除指令中非必要的背景提示，同时保持任务语义不变，测试上下文依赖；以成对指令检查规划与执行退化，提醒评测不要只覆盖冗长明确的表达。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`stardreaming/VagueBench`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as stardreaming/VagueBench
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 VagueBench 模型
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

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**指令少说几句，机器人还会做吗：删除指令中非必要的背景提示，同时保持任务语义不变，测试上下文依赖；以成对指令检查规划与执行退化，提醒评测不要只覆盖冗长明确的表达。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：删除指令中非必要的背景提示，同时保持任务语义不变，测试上下文依赖；以成对指令检查规划与执行退化，提醒评测不要只覆盖冗长明确的表达。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [vaguebench_arxiv_2611_03765.md](../../sources/papers/vaguebench_arxiv_2611_03765.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.03765](https://arxiv.org/abs/2611.03765)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.03765)
- [https://github.com/stardreaming/VagueBench](https://github.com/stardreaming/VagueBench)
