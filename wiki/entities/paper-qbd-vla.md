---
type: entity
project_id: qbd-vla
tags:
  - paper
  - policy-learning
  - vla
status: complete
updated: 2026-10-30
arxiv: "2610.38661"
code: https://github.com/jiuyixu25/QbD-VLA
related:
  - ../methods/reinforcement-learning.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../overview/embodied-station-2026-10-30-28-papers-technology-map.md
sources:
  - ../../sources/papers/qbd_vla_arxiv_2610_38661.md
  - ../../sources/repos/qbd_vla.md
  - ../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md
summary: "QbD-VLA（arXiv:2610.38661）：QbD-VLA 用短程闭环成功率选择混合精度，考虑模块之间的相互影响。"
---

# QbD-VLA（arXiv:2610.38661）

**QbD-VLA**（*Quantize by Doing: Task-Guided Mixed-Precision Quantization of Vision-Language-Action Models*，[arXiv:2610.38661](https://arxiv.org/abs/2610.38661)，[代码](https://github.com/jiuyixu25/QbD-VLA)）来自 [具身智能小站 2026-10-30 盘点](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)，属「奖励学习与 VLA 策略」组。

## 一句话定义

**QbD-VLA 用短程闭环成功率选择混合精度，考虑模块之间的相互影响。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QbD-VLA | Quantize by Doing: Task-Guided Mixed-Precision Quantization of Vision-Language-Action Models | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 按任务表现分配 VLA 的量化精度。仿真配置迁移到真机时，X-VLA 与 π₀.₅ 表现不同，说明量化收益需要按模型验证。
- 开源结论：**已开源**（步骤 2.5，2026-10-30）。仓库可达（git ls-remote）。
- 同期同组工作见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.38661](https://arxiv.org/abs/2610.38661) |
| **开源** | **已开源** |
| **要点** | QbD-VLA 用短程闭环成功率选择混合精度，考虑模块之间的相互影响。 |
| **文内指标** | 公众号转述：仿真配置迁移到真机时，X-VLA 与 π₀.₅ 表现不同，说明量化收益需要按模型验证。 |

## 源码运行时序图

以下为按仓库入口（`jiuyixu25/QbD-VLA`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as jiuyixu25/QbD-VLA
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 QbD-VLA 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：仿真配置迁移到真机时，X-VLA 与 π₀.₅ 表现不同，说明量化收益需要按模型验证。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**按任务表现分配 VLA 的量化精度：仿真配置迁移到真机时，X-VLA 与 π₀.₅ 表现不同，说明量化收益需要按模型验证。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：QbD-VLA 用短程闭环成功率选择混合精度，考虑模块之间的相互影响。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [reinforcement-learning](../methods/reinforcement-learning.md)
- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [embodied-station-2026-10-30-28-papers-technology-map](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)

## 参考来源

- [qbd_vla_arxiv_2610_38661.md](../../sources/papers/qbd_vla_arxiv_2610_38661.md)
- [wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)
- [arXiv:2610.38661](https://arxiv.org/abs/2610.38661)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.38661)
- [https://github.com/jiuyixu25/QbD-VLA](https://github.com/jiuyixu25/QbD-VLA)
