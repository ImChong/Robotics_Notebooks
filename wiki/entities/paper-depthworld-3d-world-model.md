---
type: entity
project_id: depthworld-3d-world-model
tags:
  - paper
  - world-model
  - depth
  - 3d
  - manipulation
status: complete
updated: 2026-10-14
arxiv: "2610.08780"
project: https://www.jaibardhan.com/depthworld
code: https://github.com/Jai2500/depthworld
related:
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
  - ./paper-meco-wam-4d-geometry-cotraining.md
  - ../overview/embodied-station-2026-10-13-34-papers-technology-map.md
sources:
  - ../../sources/papers/depthworld_3d_world_model_arxiv_2610_08780.md
  - ../../sources/repos/depthworld_3d_world_model.md
  - ../../sources/sites/depthworld-3d-world-model.md
  - ../../sources/blogs/wechat_embodied_station_34_papers_mulligan_2026-10-13.md
summary: "DepthWorld（arXiv:2610.08780）：先联合校准机器人运动学与多视角外参，再以潜变量空间拼接预测颜色和深度；同预算下优于仅颜色基线。"
---

# DepthWorld（arXiv:2610.08780）

**DepthWorld**（*DepthWorld: 3D World Model for Robot Manipulation*，[arXiv:2610.08780](https://arxiv.org/abs/2610.08780)，[项目页](https://www.jaibardhan.com/depthworld)，[代码](https://github.com/Jai2500/depthworld)）来自 [具身智能小站 2026-10-13 盘点](../../sources/blogs/wechat_embodied_station_34_papers_mulligan_2026-10-13.md)，属「世界模型与仿真」组。

## 一句话定义

**先联合校准机器人运动学与多视角外参，再以潜变量空间拼接预测颜色和深度；同预算下优于仅颜色基线。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 世界模型 |
| RGB-D | Color + Depth | 彩色与深度 |
| Extrinsics | Camera Extrinsics | 相机外参 |

## 为什么重要

- 几何一致性需要深度参与世界模型预测。
- 开源结论：**已开源**（步骤 2.5，2026-10-14）。项目页链接的 GitHub 仓库可达。
- 同期同组工作见 [失败导向采集与可组合世界动作模型：具身智能小站 10-13 期 34 篇阅读坐标](../overview/embodied-station-2026-10-13-34-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.08780](https://arxiv.org/abs/2610.08780) |
| **开源** | **已开源** |
| **要点** | 以几何一致性为重点。 |
| **文内指标** | 以原文为准。 |

## 源码运行时序图

以下为按仓库入口（`Jai2500/depthworld`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as Jai2500/depthworld
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 DepthWorld 模型
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

- 横向索引见 [失败导向采集与可组合世界动作模型：具身智能小站 10-13 期 34 篇阅读坐标](../overview/embodied-station-2026-10-13-34-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**加深度的世界模型值得在同训练预算下对照仅颜色基线。**

1. 开源边界：**已开源** — 项目页链接的 GitHub 仓库可达。
2. 核心机制：以几何一致性为重点。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [generative-world-models](../methods/generative-world-models.md)
- [world-action-models](../concepts/world-action-models.md)
- [paper-meco-wam-4d-geometry-cotraining](./paper-meco-wam-4d-geometry-cotraining.md)
- [embodied-station-2026-10-13-34-papers-technology-map](../overview/embodied-station-2026-10-13-34-papers-technology-map.md)

## 参考来源

- [depthworld_3d_world_model_arxiv_2610_08780.md](../../sources/papers/depthworld_3d_world_model_arxiv_2610_08780.md)
- [wechat_embodied_station_34_papers_mulligan_2026-10-13.md](../../sources/blogs/wechat_embodied_station_34_papers_mulligan_2026-10-13.md)
- [arXiv:2610.08780](https://arxiv.org/abs/2610.08780)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.08780)
- [项目页](https://www.jaibardhan.com/depthworld)
- [https://github.com/Jai2500/depthworld](https://github.com/Jai2500/depthworld)
