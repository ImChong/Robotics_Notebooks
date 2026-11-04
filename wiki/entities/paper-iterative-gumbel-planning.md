---
type: entity
project_id: iterative-gumbel-planning
tags:
  - paper
  - reward-model
  - policy-learning
status: complete
updated: 2026-11-04
arxiv: "2611.02236"
project: https://liushaohuai5.github.io/ez_igp/
code: https://github.com/liushaohuai5/EZ-IGP
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/iterative_gumbel_planning_arxiv_2611_02236.md
  - ../../sources/repos/iterative_gumbel_planning.md
  - ../../sources/sites/iterative-gumbel-planning.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "Iterative Gumbel Planning（arXiv:2611.02236）：用分维离散策略对齐搜索动作，减少把胜出候选压回高斯分布的误差；结合Gumbel采样与多轮搜索，在连续控制基准中比较性能和样本效率。"
---

# Iterative Gumbel Planning（arXiv:2611.02236）

**Iterative Gumbel Planning**（*Iterative Gumbel Planning for Continuous Control*，[arXiv:2611.02236](https://arxiv.org/abs/2611.02236)，[项目页](https://liushaohuai5.github.io/ez_igp/)，[代码](https://github.com/liushaohuai5/EZ-IGP)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「奖励学习与策略后训练」组。

## 一句话定义

**用分维离散策略对齐搜索动作，减少把胜出候选压回高斯分布的误差；结合Gumbel采样与多轮搜索，在连续控制基准中比较性能和样本效率。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Iterative Gumbel Planning | Iterative Gumbel Planning for Continuous Control | 本文方法 / 基准的简称 |

## 为什么重要

- 让连续控制策略与搜索候选对齐。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.02236](https://arxiv.org/abs/2611.02236) |
| **开源** | **已开源** |
| **要点** | 用分维离散策略对齐搜索动作，减少把胜出候选压回高斯分布的误差；结合Gumbel采样与多轮搜索，在连续控制基准中比较性能和样本效率。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`liushaohuai5/EZ-IGP`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as liushaohuai5/EZ-IGP
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Iterative Gumbel Planning 模型
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

**让连续控制策略与搜索候选对齐：用分维离散策略对齐搜索动作，减少把胜出候选压回高斯分布的误差；结合Gumbel采样与多轮搜索，在连续控制基准中比较性能和样本效率。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：用分维离散策略对齐搜索动作，减少把胜出候选压回高斯分布的误差；结合Gumbel采样与多轮搜索，在连续控制基准中比较性能和样本效率。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [iterative_gumbel_planning_arxiv_2611_02236.md](../../sources/papers/iterative_gumbel_planning_arxiv_2611_02236.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.02236](https://arxiv.org/abs/2611.02236)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.02236)
- [项目页](https://liushaohuai5.github.io/ez_igp/)
- [https://github.com/liushaohuai5/EZ-IGP](https://github.com/liushaohuai5/EZ-IGP)
