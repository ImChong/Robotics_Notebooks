---
type: entity
project_id: policyik
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-05
arxiv: "2611.05372"
project: https://tinaliu0123.github.io/PolicyIK/
code: https://github.com/Tinaliu0123/PolicyIK
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
  - ../overview/hub-embodied-eval-benchmark.md
sources:
  - ../../sources/papers/policyik_arxiv_2611_05372.md
  - ../../sources/repos/policyik.md
  - ../../sources/sites/policyik.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "PolicyIK（arXiv:2611.05372）：机械臂够到目标后还可能慢慢漂开。"
---

# PolicyIK（arXiv:2611.05372）

**PolicyIK**（*PolicyIK: High-Precision Inverse Kinematics for Robot Arms with Reinforcement Learning*，[arXiv:2611.05372](https://arxiv.org/abs/2611.05372)，[项目页](https://tinaliu0123.github.io/PolicyIK/)，[代码](https://github.com/Tinaliu0123/PolicyIK)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「策略泛化与评测」组。

## 一句话定义

**机械臂够到目标后还可能慢慢漂开。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PolicyIK | PolicyIK: High-Precision Inverse Kinematics for Robot Arms with Reinforcement Learning | 本文方法 / 基准的简称 |

## 为什么重要

- 机械臂到位后，还能稳稳停住吗。PolicyIK 让策略记住关节指令与实际位置的差，再按剩余误差调整动作幅度。
- 开源结论：**已开源**（步骤 2.5，2026-11-05）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.05372](https://arxiv.org/abs/2611.05372) |
| **开源** | **已开源** |
| **要点** | 机械臂够到目标后还可能慢慢漂开。 |
| **文内指标** | 公众号转述：评测取最后一秒的平均误差，把“停得稳”也算进成绩。 |

## 源码运行时序图

以下为按仓库入口（`Tinaliu0123/PolicyIK`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as Tinaliu0123/PolicyIK
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 PolicyIK 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：评测取最后一秒的平均误差，把“停得稳”也算进成绩。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**机械臂到位后，还能稳稳停住吗：评测取最后一秒的平均误差，把“停得稳”也算进成绩。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：机械臂够到目标后还可能慢慢漂开。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)
- [具身评测基准选型闭环（知识链）](../overview/hub-embodied-eval-benchmark.md)

## 参考来源

- [policyik_arxiv_2611_05372.md](../../sources/papers/policyik_arxiv_2611_05372.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.05372](https://arxiv.org/abs/2611.05372)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.05372)
- [项目页](https://tinaliu0123.github.io/PolicyIK/)
- [https://github.com/Tinaliu0123/PolicyIK](https://github.com/Tinaliu0123/PolicyIK)
