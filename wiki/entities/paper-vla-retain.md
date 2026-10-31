---
type: entity
project_id: vla-retain
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-10-31
arxiv: "2610.27698"
code: https://github.com/BoruiXu/VLA_vision_tuning_benchmark
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-31-50-papers-technology-map.md
sources:
  - ../../sources/papers/vla_retain_arxiv_2610_27698.md
  - ../../sources/repos/vla_retain.md
  - ../../sources/blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md
summary: "VLA-Retain（arXiv:2610.27698）：换相机或降噪后的成功率变高，仍可能牺牲其他场景。"
---

# VLA-Retain（arXiv:2610.27698）

**VLA-Retain**（*VLA-Retain: Benchmarking Cross-Dimension Effects of Vision Tuning in VLAs*，[arXiv:2610.27698](https://arxiv.org/abs/2610.27698)，[代码](https://github.com/BoruiXu/VLA_vision_tuning_benchmark)）来自 [具身智能小站 2026-10-31 盘点](../../sources/blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md)，属「评测、基准与可靠性」组。

## 一句话定义

**换相机或降噪后的成功率变高，仍可能牺牲其他场景。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA-Retain | VLA-Retain: Benchmarking Cross-Dimension Effects of Vision Tuning in VLAs | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 适配一个视角，会不会损失其他能力。VLA-Retain用同一模型调优前后的配对回合，把目标收益和非目标退化分开，适合部署验收与适配方案选择。
- 开源结论：**已开源**（步骤 2.5，2026-10-31）。仓库可达（git ls-remote）。
- 同期同组工作见 [单图重建仿真、VLA 适配评测与跨形态世界模型：具身智能小站 10-31 期 50 篇阅读坐标](../overview/embodied-station-2026-10-31-50-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.27698](https://arxiv.org/abs/2610.27698) |
| **开源** | **已开源** |
| **要点** | 换相机或降噪后的成功率变高，仍可能牺牲其他场景。 |
| **文内指标** | 公众号转述：五种VLA在七类扰动、每配置10,030回合上比较；十个模型—适配域组合中，8个出现非目标维度平均收益为负。 |

## 源码运行时序图

以下为按仓库入口（`BoruiXu/VLA_vision_tuning_benchmark`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as BoruiXu/VLA_vision_tuning_benchmark
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 VLA-Retain 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：五种VLA在七类扰动、每配置10,030回合上比较；十个模型—适配域组合中，8个出现非目标维度平均收益为负。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [单图重建仿真、VLA 适配评测与跨形态世界模型：具身智能小站 10-31 期 50 篇阅读坐标](../overview/embodied-station-2026-10-31-50-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**适配一个视角，会不会损失其他能力：五种VLA在七类扰动、每配置10,030回合上比较；十个模型—适配域组合中，8个出现非目标维度平均收益为负。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：换相机或降噪后的成功率变高，仍可能牺牲其他场景。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-31-50-papers-technology-map](../overview/embodied-station-2026-10-31-50-papers-technology-map.md)

## 参考来源

- [vla_retain_arxiv_2610_27698.md](../../sources/papers/vla_retain_arxiv_2610_27698.md)
- [wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md](../../sources/blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md)
- [arXiv:2610.27698](https://arxiv.org/abs/2610.27698)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.27698)
- [https://github.com/BoruiXu/VLA_vision_tuning_benchmark](https://github.com/BoruiXu/VLA_vision_tuning_benchmark)
