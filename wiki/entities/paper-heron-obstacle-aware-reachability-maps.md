---
type: entity
project_id: heron-obstacle-aware-reachability-maps
tags:
  - paper
  - planning
  - perception
status: complete
updated: 2026-10-30
arxiv: "2610.38233"
code: https://github.com/Lab-CORO/Heron
related:
  - ../tasks/locomotion.md
  - ../tasks/vision-language-navigation.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-30-28-papers-technology-map.md
sources:
  - ../../sources/papers/heron_obstacle_aware_reachability_maps_arxiv_2610_38233.md
  - ../../sources/repos/heron_obstacle_aware_reachability_maps.md
  - ../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md
summary: "HERON（arXiv:2610.38233）：HERON 用三维网络预测含障碍的可达性地图，并在嵌入式硬件上量化部署。"
---

# HERON（arXiv:2610.38233）

**HERON**（*HERON: Real-Time Obstacle-Aware Reachability Maps for Mobile Manipulators*，[arXiv:2610.38233](https://arxiv.org/abs/2610.38233)，[代码](https://github.com/Lab-CORO/Heron)）来自 [具身智能小站 2026-10-30 盘点](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)，属「规划、感知与建图」组。

## 一句话定义

**HERON 用三维网络预测含障碍的可达性地图，并在嵌入式硬件上量化部署。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HERON | HERON: Real-Time Obstacle-Aware Reachability Maps for Mobile Manipulators | 本文方法 / 基准的简称 |

## 为什么重要

- 实时预测障碍环境中的机械臂可达空间。它学习的是逆运动学生成的地图近似，边界区域误差和量化损失仍需结合用途评估。
- 开源结论：**已开源**（步骤 2.5，2026-10-30）。仓库可达（git ls-remote）。
- 同期同组工作见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.38233](https://arxiv.org/abs/2610.38233) |
| **开源** | **已开源** |
| **要点** | HERON 用三维网络预测含障碍的可达性地图，并在嵌入式硬件上量化部署。 |
| **文内指标** | 公众号转述：它学习的是逆运动学生成的地图近似，边界区域误差和量化损失仍需结合用途评估。 |

## 源码运行时序图

以下为按仓库入口（`Lab-CORO/Heron`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as Lab-CORO/Heron
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 HERON 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：它学习的是逆运动学生成的地图近似，边界区域误差和量化损失仍需结合用途评估。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**实时预测障碍环境中的机械臂可达空间：它学习的是逆运动学生成的地图近似，边界区域误差和量化损失仍需结合用途评估。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：HERON 用三维网络预测含障碍的可达性地图，并在嵌入式硬件上量化部署。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-30-28-papers-technology-map](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)

## 参考来源

- [heron_obstacle_aware_reachability_maps_arxiv_2610_38233.md](../../sources/papers/heron_obstacle_aware_reachability_maps_arxiv_2610_38233.md)
- [wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)
- [arXiv:2610.38233](https://arxiv.org/abs/2610.38233)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.38233)
- [https://github.com/Lab-CORO/Heron](https://github.com/Lab-CORO/Heron)
