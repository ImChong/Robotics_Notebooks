---
type: entity
project_id: roboworm
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-10-28
arxiv: "2610.33807"
project: https://huggingface.co/datasets/doanh25032004/vgp_datasets
code: https://github.com/ltdoanhwork/RoboWorM
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
sources:
  - ../../sources/papers/roboworm_arxiv_2610_33807.md
  - ../../sources/repos/roboworm.md
  - ../../sources/sites/roboworm.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "RoboWorM（arXiv:2610.33807）：利用运动学树和固定相机等硬件先验，分别检查单视角运动稳定与多视角几何一致性。"
---

# RoboWorM（arXiv:2610.33807）

**RoboWorM**（*RoboWorM: Evaluating movement realism, consistency and acceleration affects in Embodied Video World Models*，[arXiv:2610.33807](https://arxiv.org/abs/2610.33807)，[项目页](https://huggingface.co/datasets/doanh25032004/vgp_datasets)，[代码](https://github.com/ltdoanhwork/RoboWorM)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「评测与基准」组。

## 一句话定义

**利用运动学树和固定相机等硬件先验，分别检查单视角运动稳定与多视角几何一致性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RoboWorM | RoboWorM: Evaluating movement realism, consistency and acceleration affects in Embodied Video World Models | 本文方法 / 基准的简称 |

## 为什么重要

- 世界模型评测加入机器人几何约束。进一步审计加速后的物理质量，提醒读者画面观感不等于可部署的交互保真度。
- 开源结论：**已开源**（步骤 2.5，2026-10-28）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.33807](https://arxiv.org/abs/2610.33807) |
| **开源** | **已开源** |
| **要点** | 利用运动学树和固定相机等硬件先验，分别检查单视角运动稳定与多视角几何一致性。 |
| **文内指标** | 公众号转述：进一步审计加速后的物理质量，提醒读者画面观感不等于可部署的交互保真度。 |

## 源码运行时序图

以下为按仓库入口（`ltdoanhwork/RoboWorM`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as ltdoanhwork/RoboWorM
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 RoboWorM 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：进一步审计加速后的物理质量，提醒读者画面观感不等于可部署的交互保真度。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**世界模型评测加入机器人几何约束：进一步审计加速后的物理质量，提醒读者画面观感不等于可部署的交互保真度。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：利用运动学树和固定相机等硬件先验，分别检查单视角运动稳定与多视角几何一致性。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页可归入其 ② 世界模型预测保真度评测层：用运动学树与固定相机先验检查运动稳定与多视角几何一致性
- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [roboworm_arxiv_2610_33807.md](../../sources/papers/roboworm_arxiv_2610_33807.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.33807](https://arxiv.org/abs/2610.33807)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.33807)
- [项目页](https://huggingface.co/datasets/doanh25032004/vgp_datasets)
- [https://github.com/ltdoanhwork/RoboWorM](https://github.com/ltdoanhwork/RoboWorM)
