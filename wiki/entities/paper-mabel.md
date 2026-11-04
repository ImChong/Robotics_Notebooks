---
type: entity
project_id: mabel
tags:
  - paper
  - data-collection
  - cross-embodiment
status: complete
updated: 2026-11-04
arxiv: "2611.04083"
project: https://robotmabel.github.io/website/
code: https://github.com/robotmabel/MABEL
related:
  - ../concepts/embodied-data-collection-four-layers-taxonomy.md
  - ../tasks/teleoperation.md
  - ../methods/imitation-learning.md
  - ../tasks/bimanual-manipulation.md
  - ../tasks/teleoperation.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/mabel_arxiv_2611_04083.md
  - ../../sources/repos/mabel.md
  - ../../sources/sites/mabel.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "MABEL（arXiv:2611.04083）：将移动底盘、升降躯干、双臂和灵巧手接入统一控制接口，贯通遥操作采集与模仿学习；适合评估工作空间、集成成本与整机协同的团队。"
---

# MABEL（arXiv:2611.04083）

**MABEL**（*MABEL: An Open Anthropomorphic Mobile Bimanual Platform for Teleoperation and Learning*，[arXiv:2611.04083](https://arxiv.org/abs/2611.04083)，[项目页](https://robotmabel.github.io/website/)，[代码](https://github.com/robotmabel/MABEL)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「数据采集与跨形态迁移」组。

## 一句话定义

**将移动底盘、升降躯干、双臂和灵巧手接入统一控制接口，贯通遥操作采集与模仿学习；适合评估工作空间、集成成本与整机协同的团队。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MABEL | MABEL: An Open Anthropomorphic Mobile Bimanual Platform for Teleoperation and Learning | 本文方法 / 基准的简称 |

## 为什么重要

- 把双臂灵巧操作扩展到移动平台。
- 开源结论：**部分开源**（步骤 2.5，2026-11-04）。公众号代码 robotmabel/MABEL 与项目页链接的 robotmabel/open_mabel 经 git ls-remote 均失败（可能私有或尚未发布），不能当作已开源。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.04083](https://arxiv.org/abs/2611.04083) |
| **开源** | **部分开源** |
| **要点** | 将移动底盘、升降躯干、双臂和灵巧手接入统一控制接口，贯通遥操作采集与模仿学习；适合评估工作空间、集成成本与整机协同的团队。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`robotmabel/MABEL`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as robotmabel/MABEL
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 MABEL 模型
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

**把双臂灵巧操作扩展到移动平台：将移动底盘、升降躯干、双臂和灵巧手接入统一控制接口，贯通遥操作采集与模仿学习；适合评估工作空间、集成成本与整机协同的团队。**

1. 开源边界：**部分开源** — 公众号代码 robotmabel/MABEL 与项目页链接的 robotmabel/open_mabel 经 git ls-remote 均失败（可能私有或尚未发布），不能当作已开源。
2. 核心机制：将移动底盘、升降躯干、双臂和灵巧手接入统一控制接口，贯通遥操作采集与模仿学习；适合评估工作空间、集成成本与整机协同的团队。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [embodied-data-collection-four-layers-taxonomy](../concepts/embodied-data-collection-four-layers-taxonomy.md)
- [teleoperation](../tasks/teleoperation.md)
- [imitation-learning](../methods/imitation-learning.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [teleoperation](../tasks/teleoperation.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [mabel_arxiv_2611_04083.md](../../sources/papers/mabel_arxiv_2611_04083.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.04083](https://arxiv.org/abs/2611.04083)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.04083)
- [项目页](https://robotmabel.github.io/website/)
- [https://github.com/robotmabel/MABEL](https://github.com/robotmabel/MABEL)
