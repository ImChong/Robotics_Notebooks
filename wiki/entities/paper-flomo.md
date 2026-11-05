---
type: entity
project_id: flomo
tags:
  - paper
  - world-model
  - control
status: complete
updated: 2026-11-05
arxiv: "2611.05062"
project: https://flomo-wam.github.io/
code: https://github.com/pairlab/flomo
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/flomo_arxiv_2611_05062.md
  - ../../sources/repos/flomo.md
  - ../../sources/sites/flomo.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "FloMo（arXiv:2611.05062）：FloMo 让模型预测场景中各点的三维位移和机器动作。"
---

# FloMo（arXiv:2611.05062）

**FloMo**（*FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video*，[arXiv:2611.05062](https://arxiv.org/abs/2611.05062)，[项目页](https://flomo-wam.github.io/)，[代码](https://github.com/pairlab/flomo)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「世界模型与控制」组。

## 一句话定义

**FloMo 让模型预测场景中各点的三维位移和机器动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FloMo | FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video | 本文方法 / 基准的简称 |
| 3D | 3D | 三维 |

## 为什么重要

- 学人手动作时，先预测桌上哪些点会动。作者在真实操作中用同一模型比较预测视频、预测运动和两者都预测，发现加入画面预测反而拉低了策略成功率。
- 开源结论：**已开源**（步骤 2.5，2026-11-05）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.05062](https://arxiv.org/abs/2611.05062) |
| **开源** | **已开源** |
| **要点** | FloMo 让模型预测场景中各点的三维位移和机器动作。 |
| **文内指标** | 公众号转述：作者在真实操作中用同一模型比较预测视频、预测运动和两者都预测，发现加入画面预测反而拉低了策略成功率。 |

## 源码运行时序图

以下为按仓库入口（`pairlab/flomo`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as pairlab/flomo
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 FloMo 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：作者在真实操作中用同一模型比较预测视频、预测运动和两者都预测，发现加入画面预测反而拉低了策略成功率。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**学人手动作时，先预测桌上哪些点会动：作者在真实操作中用同一模型比较预测视频、预测运动和两者都预测，发现加入画面预测反而拉低了策略成功率。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：FloMo 让模型预测场景中各点的三维位移和机器动作。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [flomo_arxiv_2611_05062.md](../../sources/papers/flomo_arxiv_2611_05062.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.05062](https://arxiv.org/abs/2611.05062)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.05062)
- [项目页](https://flomo-wam.github.io/)
- [https://github.com/pairlab/flomo](https://github.com/pairlab/flomo)
