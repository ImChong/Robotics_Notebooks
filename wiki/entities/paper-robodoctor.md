---
type: entity
project_id: robodoctor
tags:
  - paper
  - vla
  - policy-learning
status: complete
updated: 2026-10-28
arxiv: "2610.35604"
project: https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo
code: https://github.com/THUSI-Lab/RoboDoctor
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
sources:
  - ../../sources/papers/robodoctor_arxiv_2610_35604.md
  - ../../sources/repos/robodoctor.md
  - ../../sources/sites/robodoctor.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "RoboDoctor（arXiv:2610.35604）：围绕控制权分配、干预时机与操作方法记录诊断、尝试和结果，支持固定权重下的执行调整。"
---

# RoboDoctor（arXiv:2610.35604）

**RoboDoctor**（*RoboDoctor: Training-Free Recursive Self-Improvement for Robot Policies*，[arXiv:2610.35604](https://arxiv.org/abs/2610.35604)，[项目页](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo)，[代码](https://github.com/THUSI-Lab/RoboDoctor)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「VLA 与策略后训练」组。

## 一句话定义

**围绕控制权分配、干预时机与操作方法记录诊断、尝试和结果，支持固定权重下的执行调整。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RoboDoctor | RoboDoctor: Training-Free Recursive Self-Improvement for Robot Policies | 本文方法 / 基准的简称 |

## 为什么重要

- 不改权重，利用执行经验修正决策。现有任务档案显示可行性，但跨回合经验的独立贡献仍需受控实验。
- 开源结论：**已开源**（步骤 2.5，2026-10-28）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.35604](https://arxiv.org/abs/2610.35604) |
| **开源** | **已开源** |
| **要点** | 围绕控制权分配、干预时机与操作方法记录诊断、尝试和结果，支持固定权重下的执行调整。 |
| **文内指标** | 公众号转述：现有任务档案显示可行性，但跨回合经验的独立贡献仍需受控实验。 |

## 源码运行时序图

以下为按仓库入口（`THUSI-Lab/RoboDoctor`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as THUSI-Lab/RoboDoctor
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 RoboDoctor 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：现有任务档案显示可行性，但跨回合经验的独立贡献仍需受控实验。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**不改权重，利用执行经验修正决策：现有任务档案显示可行性，但跨回合经验的独立贡献仍需受控实验。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：围绕控制权分配、干预时机与操作方法记录诊断、尝试和结果，支持固定权重下的执行调整。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [robodoctor_arxiv_2610_35604.md](../../sources/papers/robodoctor_arxiv_2610_35604.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.35604](https://arxiv.org/abs/2610.35604)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.35604)
- [项目页](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo)
- [https://github.com/THUSI-Lab/RoboDoctor](https://github.com/THUSI-Lab/RoboDoctor)
