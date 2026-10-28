---
type: entity
project_id: open-world-robot-autonomy
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-10-28
arxiv: "2610.34649"
project: https://huggingface.co/AetherLabs-AI/InternVLA-A1.5-RAB
code: https://github.com/AetherLabsAI/RobotAgentBench
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
sources:
  - ../../sources/papers/open_world_robot_autonomy_arxiv_2610_34649.md
  - ../../sources/repos/open_world_robot_autonomy.md
  - ../../sources/sites/open-world-robot-autonomy.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "Open-world Robot Autonomy（arXiv:2610.34649）：构建家庭布局与分层任务，覆盖任务发现、上下文推理、操作、恢复和鲁棒性，并以开放场景检验组合能力。"
---

# Open-world Robot Autonomy（arXiv:2610.34649）

**Open-world Robot Autonomy**（*Towards Open-world Robot Autonomy*，[arXiv:2610.34649](https://arxiv.org/abs/2610.34649)，[项目页](https://huggingface.co/AetherLabs-AI/InternVLA-A1.5-RAB)，[代码](https://github.com/AetherLabsAI/RobotAgentBench)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「评测与基准」组。

## 一句话定义

**构建家庭布局与分层任务，覆盖任务发现、上下文推理、操作、恢复和鲁棒性，并以开放场景检验组合能力。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Open-world Robot Autonomy | Towards Open-world Robot Autonomy | 本文方法 / 基准的简称 |

## 为什么重要

- 开放家庭环境需要综合自主性评测。帮助区分单项任务成功与持续满足高层需求的自主执行。
- 开源结论：**已开源**（步骤 2.5，2026-10-28）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.34649](https://arxiv.org/abs/2610.34649) |
| **开源** | **已开源** |
| **要点** | 构建家庭布局与分层任务，覆盖任务发现、上下文推理、操作、恢复和鲁棒性，并以开放场景检验组合能力。 |
| **文内指标** | 公众号转述：帮助区分单项任务成功与持续满足高层需求的自主执行。 |

## 源码运行时序图

以下为按仓库入口（`AetherLabsAI/RobotAgentBench`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as AetherLabsAI/RobotAgentBench
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Open-world Robot Autonomy 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：帮助区分单项任务成功与持续满足高层需求的自主执行。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**开放家庭环境需要综合自主性评测：帮助区分单项任务成功与持续满足高层需求的自主执行。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：构建家庭布局与分层任务，覆盖任务发现、上下文推理、操作、恢复和鲁棒性，并以开放场景检验组合能力。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [open_world_robot_autonomy_arxiv_2610_34649.md](../../sources/papers/open_world_robot_autonomy_arxiv_2610_34649.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.34649](https://arxiv.org/abs/2610.34649)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.34649)
- [项目页](https://huggingface.co/AetherLabs-AI/InternVLA-A1.5-RAB)
- [https://github.com/AetherLabsAI/RobotAgentBench](https://github.com/AetherLabsAI/RobotAgentBench)
