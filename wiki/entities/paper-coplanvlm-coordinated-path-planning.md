---
type: entity
project_id: coplanvlm-coordinated-path-planning
tags:
  - paper
  - vlm
  - multi-robot
  - path-planning
  - turtlebot
status: complete
updated: 2026-10-14
arxiv: "2610.14341"
code: https://github.com/AICPS/CoPlanVLM
related:
  - ../methods/marl.md
  - ../tasks/zero-shot-object-navigation.md
  - ../overview/embodied-station-2026-10-14-41-papers-technology-map.md
sources:
  - ../../sources/papers/coplanvlm_coordinated_path_planning_arxiv_2610_14341.md
  - ../../sources/repos/coplanvlm_coordinated_path_planning.md
  - ../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md
summary: "CoPlanVLM（arXiv:2610.14341）：结合自由空间叠图、任务分类与专用推理提示，把语言指令转为双机器人协同路径；Gazebo 与 Turtlebot 4 验证。"
---

# CoPlanVLM（arXiv:2610.14341）

**CoPlanVLM**（*CoPlanVLM: Coordinated Path Planning for a Robot Team using Vision Language Models*，[arXiv:2610.14341](https://arxiv.org/abs/2610.14341)，[代码](https://github.com/AICPS/CoPlanVLM)）来自 [具身智能小站 2026-10-14 盘点](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)，属「多机器人协作与评测」组。

## 一句话定义

**结合自由空间叠图、任务分类与专用推理提示，把语言指令转为双机器人协同路径；Gazebo 与 Turtlebot 4 验证。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉语言模型 |
| MRPP | Multi-Robot Path Planning | 多机器人路径规划 |
| Gazebo | Gazebo simulator | ROS 仿真器 |

## 为什么重要

- 用 VLM 做多机器人路径需要显式空间表示。
- 开源结论：**已开源**（步骤 2.5，2026-10-14）。仓库可达。
- 同期同组工作见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.14341](https://arxiv.org/abs/2610.14341) |
| **开源** | **已开源** |
| **要点** | 路径长度比较仅针对双方均成功的试验。 |
| **文内指标** | 以原文为准。 |

## 源码运行时序图

以下为按仓库入口（`AICPS/CoPlanVLM`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as AICPS/CoPlanVLM
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 CoPlanVLM 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 以原文为准。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**VLM 规划要配空间叠图与明确失败统计口径。**

1. 开源边界：**已开源** — 仓库可达。
2. 核心机制：路径长度比较仅针对双方均成功的试验。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [marl](../methods/marl.md)
- [zero-shot-object-navigation](../tasks/zero-shot-object-navigation.md)
- [embodied-station-2026-10-14-41-papers-technology-map](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)

## 参考来源

- [coplanvlm_coordinated_path_planning_arxiv_2610_14341.md](../../sources/papers/coplanvlm_coordinated_path_planning_arxiv_2610_14341.md)
- [wechat_embodied_station_41_papers_streamingwam_2026-10-14.md](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- [arXiv:2610.14341](https://arxiv.org/abs/2610.14341)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.14341)
- [https://github.com/AICPS/CoPlanVLM](https://github.com/AICPS/CoPlanVLM)
