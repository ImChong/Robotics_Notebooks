---
type: entity
project_id: just-between-us
tags:
  - paper
  - humanoid
  - tutorial
status: complete
updated: 2026-11-04
arxiv: "2611.04692"
code: https://github.com/atikkhannilgar/Discreet_Interaction
related:
  - ../tasks/locomotion.md
  - ../concepts/motion-retargeting.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/just_between_us_arxiv_2611_04692.md
  - ../../sources/repos/just_between_us.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "Just Between Us（arXiv:2611.04692）：通过耳内音频与侧倾头部建立定向交流通道，研究共享游戏中的可用性与公平感；实验采用研究员操控的屏幕代理，不能等同自主机器人。"
---

# Just Between Us（arXiv:2611.04692）

**Just Between Us**（*Just Between Us? Exploring a Wearable-Mediated Channel for Discreet Human-Agent Interaction in a Shared Game Setting*，[arXiv:2611.04692](https://arxiv.org/abs/2611.04692)，[代码](https://github.com/atikkhannilgar/Discreet_Interaction)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「人形运动与教程」组。

## 一句话定义

**通过耳内音频与侧倾头部建立定向交流通道，研究共享游戏中的可用性与公平感；实验采用研究员操控的屏幕代理，不能等同自主机器人。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Just Between Us | Just Between Us? Exploring a Wearable-Mediated Channel for Discreet Human-Agent Interaction in a Shared Game Setting | 本文方法 / 基准的简称 |

## 为什么重要

- 共享空间里的机器人私密交流。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。仓库可达（git ls-remote）。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.04692](https://arxiv.org/abs/2611.04692) |
| **开源** | **已开源** |
| **要点** | 通过耳内音频与侧倾头部建立定向交流通道，研究共享游戏中的可用性与公平感；实验采用研究员操控的屏幕代理，不能等同自主机器人。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`atikkhannilgar/Discreet_Interaction`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as atikkhannilgar/Discreet_Interaction
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Just Between Us 模型
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

**共享空间里的机器人私密交流：通过耳内音频与侧倾头部建立定向交流通道，研究共享游戏中的可用性与公平感；实验采用研究员操控的屏幕代理，不能等同自主机器人。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：通过耳内音频与侧倾头部建立定向交流通道，研究共享游戏中的可用性与公平感；实验采用研究员操控的屏幕代理，不能等同自主机器人。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [motion-retargeting](../concepts/motion-retargeting.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [just_between_us_arxiv_2611_04692.md](../../sources/papers/just_between_us_arxiv_2611_04692.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.04692](https://arxiv.org/abs/2611.04692)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.04692)
- [https://github.com/atikkhannilgar/Discreet_Interaction](https://github.com/atikkhannilgar/Discreet_Interaction)
