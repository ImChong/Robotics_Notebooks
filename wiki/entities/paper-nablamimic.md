---
type: entity
project_id: nablamimic
tags:
  - paper
  - humanoid
  - dexterous
status: complete
updated: 2026-11-03
arxiv: "2610.41654"
project: https://oamics.github.io/NablaMimic/
code: https://github.com/OamIcs/NablaMimic
related:
  - ../tasks/locomotion.md
  - ../concepts/motion-retargeting.md
  - ../concepts/dexterous-kinematics.md
  - ../overview/embodied-station-2026-11-03-49-papers-technology-map.md
sources:
  - ../../sources/papers/nablamimic_arxiv_2610_41654.md
  - ../../sources/repos/nablamimic.md
  - ../../sources/sites/nablamimic.md
  - ../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md
summary: "∇Mimic（arXiv:2610.41654）：∇Mimic 用逐样本梯度裁剪和动作目标回归缓解接触梯度不稳定，配合专用可微引擎提升训练效率。"
---

# ∇Mimic（arXiv:2610.41654）

**∇Mimic**（*∇Mimic: Ultra-Efficient Humanoid Control with Differentiable Simulation*，[arXiv:2610.41654](https://arxiv.org/abs/2610.41654)，[项目页](https://oamics.github.io/NablaMimic/)，[代码](https://github.com/OamIcs/NablaMimic)）来自 [具身智能小站 2026-11-03 盘点](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)，属「人形与灵巧控制」组。

## 一句话定义

**∇Mimic 用逐样本梯度裁剪和动作目标回归缓解接触梯度不稳定，配合专用可微引擎提升训练效率。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ∇Mimic | ∇Mimic: Ultra-Efficient Humanoid Control with Differentiable Simulation | 本文方法 / 基准的简称 |

## 为什么重要

- 让接触动力学梯度服务控制学习。应结合指定机器人和动作跟踪任务理解收益。
- 开源结论：**已开源**（步骤 2.5，2026-11-03）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.41654](https://arxiv.org/abs/2610.41654) |
| **开源** | **已开源** |
| **要点** | ∇Mimic 用逐样本梯度裁剪和动作目标回归缓解接触梯度不稳定，配合专用可微引擎提升训练效率。 |
| **文内指标** | 公众号转述：应结合指定机器人和动作跟踪任务理解收益。 |

## 源码运行时序图

以下为按仓库入口（`OamIcs/NablaMimic`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as OamIcs/NablaMimic
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 ∇Mimic 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：应结合指定机器人和动作跟踪任务理解收益。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**让接触动力学梯度服务控制学习：应结合指定机器人和动作跟踪任务理解收益。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：∇Mimic 用逐样本梯度裁剪和动作目标回归缓解接触梯度不稳定，配合专用可微引擎提升训练效率。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [motion-retargeting](../concepts/motion-retargeting.md)
- [dexterous-kinematics](../concepts/dexterous-kinematics.md)
- [embodied-station-2026-11-03-49-papers-technology-map](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)

## 参考来源

- [nablamimic_arxiv_2610_41654.md](../../sources/papers/nablamimic_arxiv_2610_41654.md)
- [wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)
- [arXiv:2610.41654](https://arxiv.org/abs/2610.41654)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.41654)
- [项目页](https://oamics.github.io/NablaMimic/)
- [https://github.com/OamIcs/NablaMimic](https://github.com/OamIcs/NablaMimic)
